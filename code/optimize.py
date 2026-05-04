"""
Ex-Situ 멤리스터 파라미터 전체 탐색
- step2에서 학습된 가중치를 로드 → patch_model → 평가만 반복
- 학습 없이 평가만 하므로 빠름 (조합당 ~10초)
- 4단계 순차 탐색: A(타일+ADC) → B(변동성) → C(물리파라미터) → D(고급)
- 이어서 실행 지원
"""
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report
import memtorch
from memtorch.bh.StochasticParameter import StochasticParameter
from model import WaferCNN
import csv
import os

# ============================================================
RESULTS_CSV = "results/optimization.csv"
NUM_REPEATS = 3
os.makedirs("results", exist_ok=True)

# 데이터 로드
data = torch.load("data/wafer_data_64.pth", weights_only=False)
X_test, y_test, classes = data['X_test'], data['y_test'], data['classes']
print(f"Test: {len(X_test)} | Classes: {len(classes)}")

# ============================================================
# Ex-Situ 평가 함수
# ============================================================
def evaluate(config, trial_id=0):
    torch.manual_seed(42 + trial_id)

    model = WaferCNN(num_classes=len(classes))
    model.load_state_dict(torch.load("software_model.pth", weights_only=True))

    tile = eval(config['tile_shape']) if config['tile_shape'] != 'None' else None
    adc = int(config['ADC_resolution']) if config.get('ADC_resolution', 'None') != 'None' else None
    quant = config.get('quant_method')
    if quant == 'None': quant = None

    r_on = float(config.get('r_on', 50))
    r_off = float(config.get('r_off', 1000))
    r_on_v = float(config.get('r_on_var', 0.0))
    r_off_v = float(config.get('r_off_var', 0.0))

    params = {
        'r_on': StochasticParameter(loc=r_on, scale=r_on * r_on_v) if r_on_v > 0 else r_on,
        'r_off': StochasticParameter(loc=r_off, scale=r_off * r_off_v) if r_off_v > 0 else r_off,
    }

    kw = {'memristor_model': memtorch.bh.memristor.VTEAM, 'memristor_model_params': params}
    if tile: kw['tile_shape'] = tile
    if adc: kw['ADC_resolution'] = adc
    if quant: kw['quant_method'] = quant

    scheme = config.get('scheme')
    if scheme and scheme != 'None':
        kw['scheme'] = memtorch.bh.Scheme.SingleColumn if 'Single' in str(scheme) else memtorch.bh.Scheme.DoubleColumn

    trans = config.get('transistor')
    if trans is not None and trans != 'None':
        kw['transistor'] = trans if isinstance(trans, bool) else trans == 'True'

    max_v = config.get('max_voltage')
    if max_v and max_v != 'None':
        kw['max_input_voltage'] = float(max_v)

    prog = config.get('programming')
    if prog and prog != 'None':
        kw['programming_routine'] = memtorch.bh.crossbar.gen_programming_signal
        tol = config.get('prog_tol')
        if tol: kw['programming_routine_params'] = {'rel_tol': float(tol)}

    patched = memtorch.patch_model(model, **kw)
    patched.eval()

    preds, labels = [], []
    with torch.no_grad():
        for x, y in DataLoader(TensorDataset(X_test, y_test), batch_size=32):
            pred = (torch.sigmoid(patched(x)) > 0.5).int()
            preds.extend(pred.cpu().numpy())
            labels.extend(y.cpu().numpy())

    rpt = classification_report(np.array(labels), np.array(preds),
                                target_names=classes, zero_division=0, output_dict=True)
    return {k: rpt['macro avg'][k] for k in ['f1-score', 'precision', 'recall']}

# ============================================================
# CSV 기록 + 이어서 실행
# ============================================================
csv_header = ['stage','trial','repeat','tile_shape','ADC_resolution','quant_method',
              'r_on','r_off','r_on_var','r_off_var','scheme','transistor',
              'max_voltage','programming','prog_tol','f1','precision','recall']

done = set()
if os.path.exists(RESULTS_CSV):
    with open(RESULTS_CSV, 'r') as f:
        next(csv.reader(f), None)
        for row in csv.reader(f):
            if len(row) >= 3: done.add((row[0], row[1], row[2]))
    print(f"Resume: {len(done)} done")
else:
    with open(RESULTS_CSV, 'w', newline='') as f:
        csv.writer(f).writerow(csv_header)

def log(stage, trial, repeat, config, result):
    with open(RESULTS_CSV, 'a', newline='') as f:
        csv.writer(f).writerow([
            stage, trial, repeat,
            config.get('tile_shape',''), config.get('ADC_resolution',''),
            config.get('quant_method',''), config.get('r_on',50), config.get('r_off',1000),
            config.get('r_on_var',0), config.get('r_off_var',0),
            config.get('scheme',''), config.get('transistor',''),
            config.get('max_voltage',''), config.get('programming',''),
            config.get('prog_tol',''),
            f"{result['f1-score']:.4f}", f"{result['precision']:.4f}", f"{result['recall']:.4f}"
        ])

def get_done_f1(stage, trial):
    f1s = []
    if os.path.exists(RESULTS_CSV):
        with open(RESULTS_CSV, 'r') as f:
            next(csv.reader(f), None)
            for row in csv.reader(f):
                if len(row) >= 16 and row[0] == stage and row[1] == str(trial):
                    f1s.append(float(row[15]))
    return f1s

def search(name, configs):
    best_f1, best_cfg = -1, None
    for i, cfg in enumerate(configs):
        f1s = []
        print(f"\n  [{name}] {i+1}/{len(configs)}")
        old_f1 = get_done_f1(name, i+1)
        for r in range(NUM_REPEATS):
            key = (name, str(i+1), str(r+1))
            if key in done:
                print(f"    {r+1}/{NUM_REPEATS}: SKIP")
                continue
            try:
                res = evaluate(cfg, trial_id=i*NUM_REPEATS+r)
                f1s.append(res['f1-score'])
                log(name, i+1, r+1, cfg, res)
                print(f"    {r+1}/{NUM_REPEATS}: F1={res['f1-score']:.4f}")
            except Exception as e:
                print(f"    {r+1}/{NUM_REPEATS}: ERR {e}")
        all_f1 = old_f1 + f1s
        if all_f1:
            avg = np.mean(all_f1)
            print(f"    → Avg: {avg:.4f}")
            if avg > best_f1:
                best_f1 = avg
                best_cfg = cfg.copy()
    print(f"\n  [{name}] Best: {best_f1:.4f}")
    return best_cfg, best_f1

# ============================================================
# 탐색
# ============================================================
print("\n" + "="*50 + "\nStage A: tile + ADC\n" + "="*50)
cfgs = []
for tile in ['None', '(64,64)', '(128,128)', '(256,256)']:
    for adc in ['None', '4', '6', '8', '10', '12']:
        cfgs.append({'tile_shape': tile, 'ADC_resolution': adc,
                     'quant_method': 'linear' if adc != 'None' else 'None'})
best_a, _ = search("A", cfgs)

print("\n" + "="*50 + "\nStage B: variation\n" + "="*50)
cfgs = []
for rv in [0.0, 0.01, 0.03, 0.05, 0.10]:
    for rfv in [0.0, 0.05, 0.10, 0.15, 0.20]:
        c = best_a.copy(); c['r_on_var'] = str(rv); c['r_off_var'] = str(rfv)
        cfgs.append(c)
best_b, _ = search("B", cfgs)

print("\n" + "="*50 + "\nStage C: r_on/r_off\n" + "="*50)
cfgs = []
for ron in [50, 100, 200]:
    for roff in [500, 1000, 2000, 5000]:
        c = best_b.copy(); c['r_on'] = str(ron); c['r_off'] = str(roff)
        cfgs.append(c)
best_c, _ = search("C", cfgs)

print("\n" + "="*50 + "\nStage D: advanced\n" + "="*50)
d1 = [];
for s in ['DoubleColumn', 'SingleColumn']:
    c = best_c.copy(); c['scheme'] = s; d1.append(c)
best_d1, _ = search("D1", d1)

d2 = []
for t in [True, False]:
    c = best_d1.copy(); c['transistor'] = t; d2.append(c)
best_d2, _ = search("D2", d2)

d3 = []
for v in ['None', '0.3', '0.6', '1.0']:
    c = best_d2.copy(); c['max_voltage'] = v; d3.append(c)
best_d3, _ = search("D3", d3)

d4 = [best_d3.copy()]
d4[0]['programming'] = 'None'
for tol in [0.1, 0.05, 0.01]:
    c = best_d3.copy(); c['programming'] = 'naive_program'; c['prog_tol'] = str(tol)
    d4.append(c)
best_d4, _ = search("D4", d4)

print("\n" + "="*50 + "\nFINAL\n" + "="*50)
print(f"Config: {best_d4}")
for i in range(3):
    r = evaluate(best_d4, trial_id=9000+i)
    print(f"  Run {i+1}: F1={r['f1-score']:.4f} P={r['precision']:.4f} R={r['recall']:.4f}")
print("Done!")
