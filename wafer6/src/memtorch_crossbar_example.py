import argparse
import copy
from pathlib import Path

from memtorch.bh.memristor.VTEAM import VTEAM
from memtorch.mn import patch_model

from run_direction1 import (
    DEFAULT_DATA,
    SmallWaferCNN,
    evaluate,
    load_data,
    make_loader,
    split_data,
    train_model,
)


def main():
    parser = argparse.ArgumentParser(description="Minimal MemTorch crossbar inference example for wafer CNN")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--max-samples", type=int, default=800)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--adc-bits", type=int, default=8)
    parser.add_argument("--tile-rows", type=int, default=128)
    parser.add_argument("--tile-cols", type=int, default=128)
    args = parser.parse_args()

    x, y = load_data(args.data, args.max_samples, args.synthetic)
    x_train, x_test, y_train, y_test = split_data(x, y)
    train_loader = make_loader(x_train, y_train, args.batch_size, shuffle=True)
    test_loader = make_loader(x_test, y_test, args.batch_size, shuffle=False)

    model = SmallWaferCNN(num_classes=int(y.max()) + 1)
    train_model(model, train_loader, epochs=args.epochs)
    ideal_acc = evaluate(model, test_loader)

    memristive_model = patch_model(
        copy.deepcopy(model),
        memristor_model=VTEAM,
        memristor_model_params={},
        ADC_resolution=args.adc_bits,
        tile_shape=(args.tile_rows, args.tile_cols),
        quant_method="linear",
        use_bindings=False,
        verbose=False,
    )
    memtorch_acc = evaluate(memristive_model, test_loader)

    print("[MemTorch example] Ideal CNN vs MemTorch VTEAM crossbar")
    print(f"ideal_cnn_accuracy={ideal_acc:.4f}")
    print(
        f"memtorch_crossbar_accuracy={memtorch_acc:.4f}, "
        f"adc_bits={args.adc_bits}, tile_shape={args.tile_rows}x{args.tile_cols}"
    )


if __name__ == "__main__":
    main()
