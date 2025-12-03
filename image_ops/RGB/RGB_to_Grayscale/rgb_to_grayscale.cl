__kernel void rgb_to_grayscale(__global int* input,__global int* output, const int height, const int width)
{
    int plane = height * width;
    int grow = get_global_id(0);
    int gcol = get_global_id(1);

    int index = grow * width + gcol;

    int R = input[index];
    int G = input[index + plane];
    int B = input[index + 2 * plane];

    output[index] = (int)(0.299f * R + 0.587f * G + 0.114f * B);
}

/*
    int plane = HEIGHT * WIDTH;
    for(int h=0; h<HEIGHT; h++)
        for(int w=0; w<WIDTH; w++)
        {
            int idx = h * WIDTH + w;
            int R = input_image[idx];
            int G = input_image[idx + plane];
            int B = input_image[idx + 2 * plane];
            output_image[idx] = int(0.299f * R + 0.587f * G + 0.114f * B);
        }
*/