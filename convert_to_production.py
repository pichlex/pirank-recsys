"""Convert trained models to production formats."""

import torch
import onnx
from pathlib import Path
import subprocess
import logging

logger = logging.getLogger(__name__)


def convert_to_onnx(model_path: Path, output_path: Path, input_shape: tuple = (1, 136)) -> None:
    """Convert PyTorch model to ONNX format."""
    from pirank_recsys.models.pirank import PiRankModel
    
    # Load model
    model = PiRankModel()
    checkpoint = torch.load(model_path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(input_shape)
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["features"],
        output_names=["scores"],
        dynamic_axes={
            "features": {0: "batch_size"},
            "scores": {0: "batch_size"}
        }
    )
    
    logger.info(f"Model converted to ONNX: {output_path}")


def convert_to_tensorrt(onnx_path: Path, output_path: Path) -> None:
    """Convert ONNX model to TensorRT format."""
    try:
        import tensorrt as trt
        
        logger_trt = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger_trt)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger_trt)
        
        # Parse ONNX model
        with open(onnx_path, "rb") as model:
            if not parser.parse(model.read()):
                for error in range(parser.num_errors):
                    logger.error(parser.get_error(error))
                raise RuntimeError("Failed to parse ONNX model")
        
        # Build engine
        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 28  # 256MB
        
        engine = builder.build_engine(network, config)
        
        # Save engine
        with open(output_path, "wb") as f:
            f.write(engine.serialize())
        
        logger.info(f"Model converted to TensorRT: {output_path}")
        
    except ImportError:
        logger.warning("TensorRT not available, using trtexec command line tool")
        
        # Fallback to command line tool
        cmd = [
            "trtexec",
            f"--onnx={onnx_path}",
            f"--saveEngine={output_path}",
            "--workspace=256"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"TensorRT conversion failed: {result.stderr}")
            raise RuntimeError("TensorRT conversion failed")
        
        logger.info(f"Model converted to TensorRT: {output_path}")


def convert_model(model_path: Path, output_dir: Path) -> None:
    """Convert model to both ONNX and TensorRT formats."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert to ONNX
    onnx_path = output_dir / "model.onnx"
    convert_to_onnx(model_path, onnx_path)
    
    # Convert to TensorRT
    tensorrt_path = output_dir / "model.trt"
    convert_to_tensorrt(onnx_path, tensorrt_path)


if __name__ == "__main__":
    import fire
    fire.Fire(convert_model)
