pulsar2 build --input SmolVLM-256M-Instruct_vision.onnx \
                --config config.json \
                --output_dir build-output4 \
                --output_name SmolVLM-256M-Instruct_vision.axmodel \
                --target_hardware AX650 \
                --compiler.check 0 
