#ifndef _MY_ENCODER_H_
#define _MY_ENCODER_H_

#include <string.h>
#include <iostream>
#include <string>
#include <memory>
#include <vector>

extern "C"
{
#include "mpp_common.h"
#include "mpp_env.h"
#include "mpp_mem.h"
#include "rk_venc_ref.h"
#include "mpi_enc_utils.h"
#include "rk_mpi.h"
#include "mpp_meta.h"
#include "rk_venc_cmd.h"
#include "mpp_packet.h"
#include "rk_type.h"
};

typedef void *MppEncRefCfg;

typedef struct MpiEnc
{
    // global flow control flag
    RK_U32 frm_eos;
    RK_U32 pkt_eos;
    RK_U32 frm_pkt_cnt;
    RK_S32 frame_count;
    RK_U64 stream_size;

    // src and dst
    //    FILE *fp_input;
    //    FILE *fp_output;

    // base flow context
    MppCtx ctx;
    MppApi *mpi;
    MppEncCfg cfg;
    MppEncPrepCfg prep_cfg;
    MppEncRcCfg rc_cfg;
    MppEncCodecCfg codec_cfg;
    MppEncSliceSplit split_cfg;
    MppEncOSDPltCfg osd_plt_cfg;
    MppEncOSDPlt osd_plt;
    MppEncOSDData osd_data;
    MppEncROIRegion roi_region[3];
    MppEncROICfg roi_cfg;

    // input / output
    MppBufferGroup buf_grp;
    MppBuffer frm_buf;
    MppBuffer pkt_buf;
    MppEncSeiMode sei_mode;
    MppEncHeaderMode header_mode;

    // paramter for resource malloc
    RK_U32 width;
    RK_U32 height;
    RK_U32 hor_stride;
    RK_U32 ver_stride;
    MppFrameFormat fmt;
    MppCodingType type;
    RK_S32 num_frames;
    RK_S32 loop_times;
    // CamSource *cam_ctx;

    // resources
    size_t header_size;
    size_t frame_size;
    /* NOTE: packet buffer may overflow */
    size_t packet_size;

    RK_U32 osd_enable;
    RK_U32 osd_mode;
    RK_U32 split_mode;
    RK_U32 split_arg;

    RK_U32 user_data_enable;
    RK_U32 roi_enable;

    // rate control runtime parameter

    RK_S32 fps_in_flex;
    RK_S32 fps_in_den;
    RK_S32 fps_in_num;
    RK_S32 fps_out_flex;
    RK_S32 fps_out_den;
    RK_S32 fps_out_num;
    RK_S32 bps;
    RK_S32 bps_max;
    RK_S32 bps_min;
    RK_S32 rc_mode;
    RK_S32 gop_mode;
    RK_S32 gop_len;
    RK_S32 vi_len;
} MpiEncData;

typedef struct MpiEncArgs_t
{
    char *file_input;
    char *file_output;
    char *file_cfg;
    // dictionary          *cfg_ini;

    MppCodingType type;
    MppFrameFormat format;
    RK_S32 num_frames;
    RK_S32 loop_cnt;

    RK_S32 width;
    RK_S32 height;
    RK_S32 hor_stride;
    RK_S32 ver_stride;

    RK_S32 bps_target;
    RK_S32 bps_max;
    RK_S32 bps_min;
    RK_S32 rc_mode;

    RK_S32 fps_in_flex;
    RK_S32 fps_in_num;
    RK_S32 fps_in_den;
    RK_S32 fps_out_flex;
    RK_S32 fps_out_num;
    RK_S32 fps_out_den;

    RK_S32 gop_mode;
    RK_S32 gop_len;
    RK_S32 vi_len;

    MppEncHeaderMode header_mode;

    MppEncSliceSplit split;
} MpiEncArgs;

typedef struct MppEncCpbInfo_t
{
    RK_S32 dpb_size;
    RK_S32 max_lt_cnt;
    RK_S32 max_st_cnt;
    RK_S32 max_lt_idx;
    RK_S32 max_st_tid;
    /* loop length of st/lt config */
    RK_S32 lt_gop;
    RK_S32 st_gop;
} MppEncCpbInfo;

typedef struct MppEncRefCfgImpl_t
{
    const char *name;
    RK_S32 ready;
    RK_U32 debug;

    /* config from user */
    RK_S32 max_lt_cfg;
    RK_S32 max_st_cfg;
    RK_S32 lt_cfg_cnt;
    RK_S32 st_cfg_cnt;
    MppEncRefLtFrmCfg *lt_cfg;
    MppEncRefStFrmCfg *st_cfg;

    /* generated parameter for MppEncRefs */
    MppEncCpbInfo cpb_info;
} MppEncRefCfgImpl;

class MyEncoder
{
public:
    MyEncoder() { printf("MyEncoder init\n"); }

    ~MyEncoder() { deinit(); }
    MPP_RET encode(const void *img, int img_len, char *dst, int *length);
    void setUp(int width, int height, int fps);
    void MppEncoderInit(int width, int height, int fps) { setUp(width, height, fps); }

private:
    // ��ʼ��������Ϣ
    // �������ñ���ĸ�ʽ, ��ֱ��ˮƽƫ��������Ϣ
    void init();

    // �������ý����ǣ��ͷ���Դ
    void deinit();

    // void calcSize(int src_w, int src_h);

    // �ؼ�֡��ͷ����Ϣ д��dst��, д��ĳ���Ϊ*length
    MPP_RET WriteHeadInfo(char *dst, int *length);

    MPP_RET WriteHeadInfo(FILE *fp);

    MPP_RET enc_ctx_init(MpiEncData **data, MpiEncArgs *cmd);

    MPP_RET enc_ctx_deinit(MpiEncData **data);

    MPP_RET test_mpp_setup_legacy(MpiEncData *p);

    MPP_RET test_mpp_enc_cfg_setup(MpiEncData *p);

    MPP_RET mpi_enc_gen_osd_plt(MppEncOSDPlt *osd_plt, RK_U32 *table);

    // MPP_RET mpi_enc_gen_ref_cfg(MppEncRefCfg ref);

    MPP_RET mpi_enc_gen_osd_data(MppEncOSDData *osd_data, MppBuffer osd_buf,
                                 RK_U32 frame_cnt);

private:
    MpiEncArgs args_;
    MpiEncData *p;
    MppPacket packet;
    int countIdx_;
    int top, bottom, left, right;
    bool append;
};

#endif

