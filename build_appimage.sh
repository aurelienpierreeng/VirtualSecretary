#!/bin/bash

# Build Virtual Secretary within an AppDir directory
# Then package it as an .AppImage
# Copyright (c) Aurélien Pierre - 2023

# Build the app package
# Reference : https://python-appimage.readthedocs.io/en/latest/apps/#advanced-packaging
python-appimage build app -p 3.11 recipe/
chmod +x VirtualSecretary-x86_64.AppImage
