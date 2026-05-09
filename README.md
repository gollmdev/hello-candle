cargo new myapp
cd myapp

<!-- 从 GitHub 上的 Hugging Face Candle 仓库直接安装 candle-core 库，并启用 cuda 编译特性。 -->
cargo add --git https://github.com/huggingface/candle.git candle-core --features "cuda"

<!-- https://developer.nvidia.com/cuda-toolkit-archive -->
cargo build



https://github.com/huggingface/candle/tree/main/candle-examples/examples/llama
