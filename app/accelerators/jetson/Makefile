



benchmark_gpu:
	@echo "Compiling and running GPU version for benchmarking"
	python ./code/SampleStates.py ./lib/model_2380.h5 ./tests/sample_input.txt GPU 15
	

benchmark_cpu:
	@echo "Compiling and running CPU version for benchmarking"
	python ./code/SampleStates.py ./lib/model_2380.h5 ./tests/sample_input.txt CPU 15

test_gpu:
	@echo "Compiling and running GPU version for testing"
	python ./code/SampleStates.py ./lib/model_2380.h5 ./tests/sample_input.txt GPU 1 test	

test_cpu:
	@echo "Compiling and running CPU version for testing"
	python ./code/SampleStates.py ./lib/model_2380.h5 ./tests/sample_input.txt CPU 1 test

clean:
	@echo "CLEANING results file from test run"	
	rm -f ./tests/results.txt


help:
	@echo "make benchmark_gpu to run GPU benchmark"
	@echo "make benchmark_cpu to run CPU benchmark"
	@echo "make test_gpu to run GPU tests"
	@echo "make test_cpu to run CPU tests"
	@echo "make clean to clean temporaries"

