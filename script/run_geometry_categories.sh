#!/bin/bash

CMD=$1
YML=$2

for cat in all BeerBottle Bottle Bowl Cookie Cup DrinkBottle DrinkingUtensil Mirror Mug PillBottle Plate Ring Spoon Statue Teacup Teapot ToyFigure Vase WineBottle WineGlass
do
    yml="${YML:0:(-4)}-$cat.yml"
    cp $YML $yml
    cmd="${CMD/$YML/$yml}"
    cmd="$cmd --category $cat"
    eval $cmd
done
