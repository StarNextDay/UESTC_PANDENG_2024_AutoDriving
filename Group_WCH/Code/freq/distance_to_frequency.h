#ifndef __DISTANCE_TO_FREQUENCY_H
#define __DISTANCE_TO_FREQUENCY_H

#define dmax 80

//extern distance
int distance_to_frequency(float distance){
    if(0.8*dmax<=distance && distance < dmax)
        return 150;
    else if(0.6*dmax<=distance && distance < 0.8*dmax)
        return 300;
    else if(0.4*dmax<=distance && distance < 0.6*dmax)
        return 450;
    else if(0.2*dmax<=distance && distance < 0.4*dmax)
        return 600;
    else if(0<=distance && distance < 0.2*dmax)
        return 750;
    else 
        return 0;
}

#endif
