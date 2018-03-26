NVCC = $(shell which gcc)
NVCC_FLAGS = -O3 -g -Wall

all: main.exe

main.exe: main.o kernel.o
	$(NVCC) $^ -o $@ -lm

main.o: main.c kernel.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ -lm

kernel.o: kernel.c kernel.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ -lm

clean:
	rm -f *.o *.exe
