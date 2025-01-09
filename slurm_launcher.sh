#!/bin/bash

benchopt clean
benchopt run --slurm slurm_config_cpu.yaml --timeout 10h --no-plot