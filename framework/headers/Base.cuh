#ifndef BASE_H
#define BASE_H

class BaseLayer{
public:
    BaseLayer(bool trainable = false);

protected:
    bool trainable;
    bool test_mode;
};

#endif 