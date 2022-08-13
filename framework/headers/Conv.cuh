#ifndef CONV_H
#define CONV_H

class Conv{
public:
    Conv(int in_h, int in_w, int ker_h, int ker_w, int stride = 1, char pad = 'V');
    ~Conv();
    double* forward(double *input);

// protected:
    double *weights;
    double *bias;
    double *pad_input;
    double *output;
    int stride;
    char pad;
    int ker_h;
    int ker_w;
    int in_h;
    int in_w;
    int left_up;
    int right_down;
    int out_w;
    int out_h;
};

#endif