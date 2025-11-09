set term png size 1000, 1000
set output "tnd.png"
set title "<%= title %>"

set style function pm3d
set palette rgb 7,5,15

unset ytics
set autoscale xfix
set autoscale yfix
set autoscale cbfix

splot '-' using 1:2:3
