#!/bin/bash
rm doc.aux doc.bbl doc.bcf doc.blg doc.dvi doc.fdb_latexmk doc.fls doc.log doc.old doc.pdf doc.run.xml doc.synctex.gx doc.toc
latexmk -dvi -pdf doc.tex
biber doc.bcf
latexmk -dvi -pdf doc.tex

