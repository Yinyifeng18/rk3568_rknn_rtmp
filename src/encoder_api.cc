
#include "MyEncoder.h"

static char dst[1024 * 1024 * 4];
static char img[1024 * 1024 * 4];
static std::vector<char> buffer;
static MyEncoder myEncoder;

// ��ʼ��������
void init_encoder(int w, int h, int fps)
{
    myEncoder.MppEncoderInit(w, h, 30);
    char *pdst = dst;
    // *3 ��Ϊ��RGB��ʽ
    const size_t imageSize = w * h * 3;
    buffer.resize(imageSize);
}

// ��raw��ʽ����Ϊh264
void encode_rgb_image_to_h264(char *data, int data_len, char *endoced_data, int *encoded_data_len)
{
    myEncoder.encode(data, data_len, endoced_data, encoded_data_len);
}


