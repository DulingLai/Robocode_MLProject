# Robocode_MLProject
A Robocode Robot with neural network function approaximation.  
The robot can choose to either do wave surfing, move forward/backward or fire based on the current states.  
The neural network is now built on Encog for better learning performance.  

## Performance.
Training against most bots can be done in 1000 rounds.  
When evaluating the performance after training, better set exploration off in the source code (epsilon = 0.0;) for optimal performance.
### Win Rate:  
Against TrackFire: ~99% win rate.  
Against SpinBot: ~100% win rate.  
Against Tracker: ~99% win rate.  
Against Fire: ~100% win rate.  
...   
...  

## Usage
1) Add Encog (encog.org) to dependency.  
2) Compile project. 
3) Run project with configuration "-Dsun.io.useCanonCaches=false -Ddebug=false -DNOSECURITY=true". 
4) In "Preferences/Development Options", add "$PROJECTFOLDER/out/production/Robocode_MLProject". 
5) In "Battle/New Battle" select "bots.BasicWaveSurferBot" and any opponent to start training.
