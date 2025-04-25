#!/usr/bin/env bash

project_name='prog3_unit2_neuralnetwork_v2025_01'
source_code='
  neural_network.h
  '
rm -f ${project_name}.zip
zip -r -S ${project_name} ${source_code}