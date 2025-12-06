# Makefile for Yolah Book Project
#
# This Makefile automates the compilation of LaTeX documents
# and provides convenient targets for building, cleaning, and viewing.
#
# Usage:
#   make pdf        - Full build with bibliography
#   make quick      - Fast build without bibliography
#   make view       - Open PDF in viewer
#   make clean      - Remove build artifacts
#   make deepclean  - Remove all generated files including PDF
#   make watch      - Continuous build (requires entr)
#   make figures    - Generate figures from Python script
#   make wordcount  - Count words in document
#   make check      - Check for TODO/FIXME markers
#   make help       - Show all available targets

# Configuration
MAIN = book
MAIN_TEX = $(MAIN).tex
MAIN_PDF = $(MAIN).pdf

# LaTeX compiler settings
LATEX = pdflatex
LATEX_FLAGS = --shell-escape -interaction=nonstopmode -halt-on-error
BIBER = biber

# Python for figure generation
PYTHON = python3
FIG_SCRIPT = generate_figures.py

# PDF viewer (change based on your OS)
# Linux: xdg-open or evince
# macOS: open
# Windows: start
VIEWER = xdg-open

# Detect OS for viewer
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    VIEWER = open
endif
ifeq ($(UNAME_S),Linux)
    VIEWER = xdg-open
endif

# Colors for output
NO_COLOR=\033[0m
GREEN=\033[0;32m
YELLOW=\033[0;33m
BLUE=\033[0;34m
RED=\033[0;31m

#########################################################################
# MAIN TARGETS
#########################################################################

.PHONY: all pdf quick view clean deepclean help

# Default target
all: pdf

# Full build with bibliography
pdf: figures
	@echo "$(BLUE)Building $(MAIN_PDF) with bibliography...$(NO_COLOR)"
	@$(LATEX) $(LATEX_FLAGS) $(MAIN_TEX)
	@$(BIBER) $(MAIN)
	@$(LATEX) $(LATEX_FLAGS) $(MAIN_TEX)
	@$(LATEX) $(LATEX_FLAGS) $(MAIN_TEX)
	@echo "$(GREEN)Build complete: $(MAIN_PDF)$(NO_COLOR)"

# Quick build without bibliography (faster)
quick:
	@echo "$(BLUE)Quick build of $(MAIN_PDF)...$(NO_COLOR)"
	@$(LATEX) $(LATEX_FLAGS) $(MAIN_TEX)
	@echo "$(GREEN)Quick build complete: $(MAIN_PDF)$(NO_COLOR)"

# Open PDF in viewer
view: $(MAIN_PDF)
	@echo "$(BLUE)Opening $(MAIN_PDF)...$(NO_COLOR)"
	@$(VIEWER) $(MAIN_PDF) &

# Clean build artifacts (keep PDF)
clean:
	@echo "$(YELLOW)Cleaning build artifacts...$(NO_COLOR)"
	@rm -f *.aux *.log *.out *.toc *.bbl *.blg *.bcf *.run.xml
	@rm -f *.lof *.lot *.fls *.fdb_latexmk *.synctex.gz
	@rm -rf _minted-*
	@rm -f chapters/*.aux
	@echo "$(GREEN)Clean complete$(NO_COLOR)"

# Deep clean (remove PDF and all generated files)
deepclean: clean
	@echo "$(RED)Deep cleaning (removing PDF)...$(NO_COLOR)"
	@rm -f $(MAIN_PDF)
	@rm -f figures/*.png figures/*.pdf figures/*.jpg
	@echo "$(GREEN)Deep clean complete$(NO_COLOR)"

#########################################################################
# FIGURE GENERATION
#########################################################################

.PHONY: figures

figures:
	@if [ -f $(FIG_SCRIPT) ]; then \
		echo "$(BLUE)Generating figures...$(NO_COLOR)"; \
		$(PYTHON) $(FIG_SCRIPT); \
		echo "$(GREEN)Figures generated$(NO_COLOR)"; \
	else \
		echo "$(YELLOW)No figure generation script found ($(FIG_SCRIPT))$(NO_COLOR)"; \
	fi

#########################################################################
# CONTINUOUS BUILD
#########################################################################

.PHONY: watch

# Continuous build (requires entr: brew install entr or apt-get install entr)
watch:
	@echo "$(BLUE)Watching for changes... (Ctrl-C to stop)$(NO_COLOR)"
	@ls $(MAIN_TEX) chapters/*.tex references.bib | entr -c make quick

#########################################################################
# ANALYSIS TARGETS
#########################################################################

.PHONY: wordcount check validate

# Word count (approximate, excludes LaTeX commands)
wordcount:
	@echo "$(BLUE)Counting words...$(NO_COLOR)"
	@if [ -f $(MAIN_TEX) ]; then \
		detex $(MAIN_TEX) | wc -w; \
	else \
		texcount -inc -sum $(MAIN_TEX); \
	fi

# Check for TODO and FIXME markers
check:
	@echo "$(BLUE)Checking for TODO/FIXME markers...$(NO_COLOR)"
	@grep -rn "TODO\|FIXME" $(MAIN_TEX) chapters/ || echo "$(GREEN)No TODO/FIXME found$(NO_COLOR)"

# Validate references and citations
validate: pdf
	@echo "$(BLUE)Checking for undefined references...$(NO_COLOR)"
	@grep -i "undefined" $(MAIN).log || echo "$(GREEN)No undefined references$(NO_COLOR)"
	@echo "$(BLUE)Checking for citation warnings...$(NO_COLOR)"
	@grep -i "citation.*undefined" $(MAIN).log || echo "$(GREEN)No undefined citations$(NO_COLOR)"

#########################################################################
# DEVELOPMENT TARGETS
#########################################################################

.PHONY: draft final

# Draft mode (faster, lower quality)
draft:
	@echo "$(BLUE)Building draft version...$(NO_COLOR)"
	@$(LATEX) -draftmode $(LATEX_FLAGS) $(MAIN_TEX)
	@echo "$(GREEN)Draft build complete$(NO_COLOR)"

# Final mode (full quality, with all optimizations)
final: clean pdf
	@echo "$(GREEN)Final version built: $(MAIN_PDF)$(NO_COLOR)"
	@ls -lh $(MAIN_PDF)

#########################################################################
# MODULAR BUILD (for book_modular.tex)
#########################################################################

.PHONY: modular modular-quick

MODULAR_MAIN = book_modular
MODULAR_TEX = $(MODULAR_MAIN).tex
MODULAR_PDF = $(MODULAR_MAIN).pdf

modular: figures
	@echo "$(BLUE)Building $(MODULAR_PDF) with bibliography...$(NO_COLOR)"
	@$(LATEX) $(LATEX_FLAGS) $(MODULAR_TEX)
	@$(BIBER) $(MODULAR_MAIN)
	@$(LATEX) $(LATEX_FLAGS) $(MODULAR_TEX)
	@$(LATEX) $(LATEX_FLAGS) $(MODULAR_TEX)
	@echo "$(GREEN)Modular build complete: $(MODULAR_PDF)$(NO_COLOR)"

modular-quick:
	@echo "$(BLUE)Quick build of $(MODULAR_PDF)...$(NO_COLOR)"
	@$(LATEX) $(LATEX_FLAGS) $(MODULAR_TEX)
	@echo "$(GREEN)Quick modular build complete: $(MODULAR_PDF)$(NO_COLOR)"

#########################################################################
# UTILITY TARGETS
#########################################################################

.PHONY: setup install-deps show-log

# Setup directories
setup:
	@echo "$(BLUE)Setting up directories...$(NO_COLOR)"
	@mkdir -p figures chapters
	@echo "$(GREEN)Directories created$(NO_COLOR)"

# Install LaTeX dependencies (Ubuntu/Debian)
install-deps:
	@echo "$(BLUE)Installing LaTeX dependencies...$(NO_COLOR)"
	@echo "$(YELLOW)This requires sudo privileges$(NO_COLOR)"
	sudo apt-get update
	sudo apt-get install -y texlive-full texlive-latex-extra biber python3-pygments
	@echo "$(GREEN)Dependencies installed$(NO_COLOR)"

# Show last 50 lines of log file
show-log:
	@if [ -f $(MAIN).log ]; then \
		echo "$(BLUE)Last 50 lines of $(MAIN).log:$(NO_COLOR)"; \
		tail -n 50 $(MAIN).log; \
	else \
		echo "$(RED)Log file not found$(NO_COLOR)"; \
	fi

#########################################################################
# HELP
#########################################################################

help:
	@echo "$(BLUE)Yolah Book Project - Available Make Targets$(NO_COLOR)"
	@echo ""
	@echo "$(GREEN)Main Targets:$(NO_COLOR)"
	@echo "  make pdf          - Full build with bibliography (recommended)"
	@echo "  make quick        - Fast build without bibliography"
	@echo "  make view         - Open PDF in default viewer"
	@echo "  make clean        - Remove build artifacts (keep PDF)"
	@echo "  make deepclean    - Remove all generated files including PDF"
	@echo ""
	@echo "$(GREEN)Figure Generation:$(NO_COLOR)"
	@echo "  make figures      - Generate figures from Python script"
	@echo ""
	@echo "$(GREEN)Continuous Development:$(NO_COLOR)"
	@echo "  make watch        - Auto-rebuild on file changes (requires entr)"
	@echo "  make draft        - Quick draft build (lower quality)"
	@echo "  make final        - Clean build for final version"
	@echo ""
	@echo "$(GREEN)Modular Build (book_modular.tex):$(NO_COLOR)"
	@echo "  make modular      - Build modular version with bibliography"
	@echo "  make modular-quick- Quick build of modular version"
	@echo ""
	@echo "$(GREEN)Analysis:$(NO_COLOR)"
	@echo "  make wordcount    - Count words in document"
	@echo "  make check        - Check for TODO/FIXME markers"
	@echo "  make validate     - Check for undefined references"
	@echo "  make show-log     - Show last 50 lines of build log"
	@echo ""
	@echo "$(GREEN)Setup:$(NO_COLOR)"
	@echo "  make setup        - Create necessary directories"
	@echo "  make install-deps - Install LaTeX dependencies (Ubuntu/Debian)"
	@echo ""
	@echo "$(GREEN)Help:$(NO_COLOR)"
	@echo "  make help         - Show this help message"
	@echo ""
	@echo "$(YELLOW)Note: Default target is 'pdf'$(NO_COLOR)"

#########################################################################
# BENCHMARK TARGETS
#########################################################################

.PHONY: bench bench-clean bench-run

# Compile the benchmark
bench: magic_bench.cpp
	@echo "$(BLUE)Compiling magic_bench...$(NO_COLOR)"
	@g++ -std=c++23 -O3 -march=native -Wall -Wextra magic_bench.cpp -o magic_bench -lbenchmark -lpthread
	@echo "$(GREEN)Benchmark compiled: magic_bench$(NO_COLOR)"

# Run the benchmark
bench-run: bench
	@echo "$(BLUE)Running benchmark...$(NO_COLOR)"
	@./magic_bench

# Clean benchmark executable
bench-clean:
	@echo "$(YELLOW)Cleaning benchmark executable...$(NO_COLOR)"
	@rm -f magic_bench
	@echo "$(GREEN)Benchmark clean complete$(NO_COLOR)"

#########################################################################
# DEPENDENCIES
#########################################################################

# Rebuild if source files change
$(MAIN_PDF): $(MAIN_TEX) chapters/*.tex references.bib
	@$(MAKE) pdf
