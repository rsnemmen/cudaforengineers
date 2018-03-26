NVCC = $(shell which gcc)
NVCC_FLAGS = -O3 -g -Wall

all: main.exe

main.exe: main.o kernel.o
	$(NVCC) $^ -o $@

main.o: main.c kernel.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

kernel.o: kernel.c kernel.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

clean:
	rm -f *.o *.exe
