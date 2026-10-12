CC      = gcc
SRC     = main.c common/common.c configuration/config.c \
          layer/layer.c randomizer/randomizer.c \
          initialize/initialize.c training/training.c
LIBS    = -lm

CONFIG       = configuration/configfile.txt
CONFIG_SHORT = configuration/config_short.txt

.PHONY: all seq omp gprof prof-gprof perf prof-perf short clean

all: seq

# Full sequential binary (timing / later speedup baseline)
seq: exec
exec: $(SRC)
	$(CC) -Ofast -g $(SRC) -o exec $(LIBS)

# OpenMP binary (section 3). Threads: OMP_NUM_THREADS=N ./exec_omp
omp: exec_omp
exec_omp: $(SRC)
	$(CC) -Ofast -g -fopenmp $(SRC) -o exec_omp $(LIBS)

# gprof binary (function % for section 2.3)
gprof: exec_gprof
exec_gprof: $(SRC)
	$(CC) -O2 -pg -g $(SRC) -o exec_gprof $(LIBS)

# Run gprof with the short config, write the report
prof-gprof: exec_gprof
	./exec_gprof $(CONFIG_SHORT)
	gprof exec_gprof gmon.out > gprof_report.txt
	@echo "Wrote gprof_report.txt"

# perf record + text report (screenshot / figure)
perf: exec
prof-perf: exec
	perf record -g -o perf.data ./exec $(CONFIG_SHORT)
	perf report -i perf.data --stdio > perf_report.txt
	@echo "Wrote perf.data and perf_report.txt"

# Same as seq, run the reduced config
short: exec
	./exec $(CONFIG_SHORT)

clean:
	rm -f exec exec_omp exec_gprof *.o gmon.out perf.data gprof_report.txt perf_report.txt
