NVCC = /usr/local/cuda/bin/nvcc
NVCC_FLAGS = -g -G -Xcompiler -Wall

main.exe: kernel.cu
	$(NVCC) $(NVCC_FLAGS) $^ -o $@

cpu: cpucode.c
	gcc -O3 $^ -o $@

clean:
	rm -f *.o *.exe
