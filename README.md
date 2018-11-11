# cgpde-lib-imbalanced
Modified Version of the CGP Library
======

Hybridization of Cartesian Genetic Programming and Differential Evolution to generate Artificial Neural Networks.   
The methods are applied to imbalanced data classification problems by using different objective functions: accuracy, G-mean, F-score, and area under the ROC curve (AUC).
It includes the CGPDE-IN, CGPDE-OUT-T, and CGPDE-OUT-V methods.

Author: Johnathan M Melo Neto   
Email: jmmn.mg@gmail.com

Credits of the original work are placed below.

CGP Library
======

A cross platform Cartesian Genetic Programming Library written in C.

Author: Andrew James Turner    
Webpage: http://www.cgplibrary.co.uk/     
Email: andrew.turner@york.ac.uk    
License: Lesser General Public License (LGPL) 

If this library is used in published work I would greatly appreciate a citation to the following:  

A. J. Turner and J. F. Miller. [**Introducing A Cross Platform Open Source Cartesian Genetic Programming Library**](http://andrewjamesturner.co.uk/files/GPEM2014.pdf). The Journal of Genetic Programming and Evolvable Machines, 2014, 16, 83-91.

## To Install

### On Linux

#### From Source


First you'll want to clone the repository:

`git clone https://github.com/johnathanmelo/cgpde-lib-imbalanced.git`

Once that's finished, navigate to the Root directory. In this case it would be ./cgpde-lib-imbalanced:

`cd ./cgpde-lib-imbalanced`

Then run Makefile:

`make main`

Now you can run the algorithms by running:

`./main`
