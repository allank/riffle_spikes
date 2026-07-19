module github.com/allank/riffle_spikes

go 1.22.3

require gopkg.in/yaml.v3 v3.0.1

require github.com/allank/onnx_test/bge_bench v0.0.0-00010101000000-000000000000

require gonum.org/v1/gonum v0.15.0 // indirect

replace github.com/allank/onnx_test/bge_bench => ../onnx_test/inference/bge_bench
