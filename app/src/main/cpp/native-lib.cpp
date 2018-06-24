#include <jni.h>
#include <string>
#include <opencv2/opencv.hpp>
#include <stdlib.h>
#include <stdio.h>

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl3.h>
#include <GLES3/gl32.h>
#include <android/log.h>

#include "glm/vec3.hpp"


#define LOG(...) __android_log_print(ANDROID_LOG_DEBUG, "LYH", __VA_ARGS__);


static const char gComputeShader[] =
        "#version 320 es\n"
                "layout(local_size_x = 8) in;\n"
                "layout(binding = 0) readonly buffer Input0 {\n"
                "    uint data[];\n"
                "} input0;\n"
                "layout(binding = 1) readonly buffer Input1 {\n"
                "    uint data[];\n"
                "} input1;\n"
                "layout(binding = 2) writeonly buffer Output {\n"
                "    uint data[];\n"
                "} output0;\n"
                "void main()\n"
                "{\n"
                "    uint idx = gl_GlobalInvocationID.x;\n"
                "    uint f = input0.data[idx] + input1.data[idx];\n"
                "    uint v = uint(200);\n"
       //         "    if (  (f >  v ))\n"
                "    {\n"
                "       output0.data[idx] = f;\n"
                "     }\n"
                "}\n";



#define CHECK() \
{\
    GLenum err = glGetError(); \
    if (err != GL_NO_ERROR) \
    {\
       LOG("glGetError returns %d\n", err); \
    }\
    else \
    {\
        LOG("CHECK OK"); \
    }\
}

GLuint loadShader(GLenum shaderType, const char* pSource) {

    GLuint shader = glCreateShader(shaderType);
    if (shader) {
        glShaderSource(shader, 1, &pSource, NULL);
        glCompileShader(shader);
        GLint compiled = 0;
        glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
        if (!compiled) {
            GLint infoLen = 0;
            glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &infoLen);
            if (infoLen) {
                char* buf = (char*) malloc(infoLen);
                if (buf) {
                    glGetShaderInfoLog(shader, infoLen, NULL, buf);
                    LOG("Could not compile shader: %s ", buf);
                    free(buf);
                }
                glDeleteShader(shader);
                shader = 0;
            }
        }
    }
    return shader;
}

GLuint createComputeProgram(const char* pComputeSource) {
    GLuint computeShader = loadShader(GL_COMPUTE_SHADER, pComputeSource);
    if (!computeShader) {
        return 0;
    }

    GLuint program = glCreateProgram();
    if (program) {
        glAttachShader(program, computeShader);
        glLinkProgram(program);
        GLint linkStatus = GL_FALSE;
        glGetProgramiv(program, GL_LINK_STATUS, &linkStatus);
        if (linkStatus != GL_TRUE) {
            GLint bufLength = 0;
            glGetProgramiv(program, GL_INFO_LOG_LENGTH, &bufLength);
            if (bufLength) {
                char* buf = (char*) malloc(bufLength);
                if (buf) {
                    glGetProgramInfoLog(program, bufLength, NULL, buf);
                    LOG("Could not link program: %s", buf)
                    free(buf);
                }
            }
            glDeleteProgram(program);
            program = 0;
        }
    }
    return program;
}

void setupSSBufferObject(GLuint& ssbo, GLuint index, unsigned  char * pIn, GLuint count)
{
    glGenBuffers(1, &ssbo);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);

    glBufferData(GL_SHADER_STORAGE_BUFFER, count * sizeof(unsigned char), pIn, GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, index, ssbo);
}

//测试使用compute shader来生成一张新的图片
void myImageStoreTest()
{

    const char gComputeShaderSrc2[] = {
            "#version 320 es\n"
                    "layout(local_size_x= 10, local_size_y = 10) in;\n"
                    "layout(binding = 0, rgba8ui) writeonly uniform uimage2D inimage;\n"
                    "void main(){\n"
                    "   imageStore(inimage, ivec2(gl_GlobalInvocationID.xy), uvec4(255, 255, 0, 255));\n"
                    "}"
    };


    LOG("start");
    int w = 400, h = 300;

    GLuint computeProgram;
    CHECK();
    computeProgram = createComputeProgram(gComputeShaderSrc2);
    LOG("create program end");
    CHECK();
    glUseProgram(computeProgram);

    GLuint texture;
    glGenTextures(1, &texture);
    //glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8UI, w, h);//
    glDepthMask(GL_FALSE);
    glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
    glBindImageTexture(0, texture, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA8UI);
    //glUseProgram(computeProgram);
    CHECK();
    glDispatchCompute(40, 30, 1);   // arraySize/local_size_x
    CHECK();
    LOG("dispatch end");

    struct timespec slptm;
    slptm.tv_sec = 2;
    slptm.tv_nsec = 1000;      //1000 ns = 1 us
    if (nanosleep(&slptm, NULL) == -1) {
        perror("Failed to nanosleep");
    }

    glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);
    CHECK();
    if(false) {
        glBindTexture(GL_TEXTURE_2D, texture);
        uint *outdata = (uint *) glMapBufferRange(GL_TEXTURE_2D, 0, 400 * 300 * sizeof(int) / 4,
                                                  GL_MAP_READ_BIT);

        CHECK();
        LOG("%d %d %d %d", outdata[0], outdata[1], outdata[2], outdata[3]);
        glUnmapBuffer(GL_TEXTURE_BUFFER);
    }
    if(true){
        GLuint  pbo;
        glGenBuffers(1, &pbo);
        glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo);
        glBufferData(GL_PIXEL_PACK_BUFFER, w*h* 4, NULL, GL_STREAM_READ);

        glBindTexture(GL_TEXTURE_2D, texture);

    }
    glDeleteProgram(computeProgram);
    CHECK();
    LOG("end...");
}


void yuvToRgb()
{
    //   1  |   2
    //   3  |   4
    const char src[] = {
            "#version 320 es\n"
            "layout(local_size_x = 1, local_size_y = 1) in;\n"
            "layout(binding = 0) readonly buffer DataY {\n"
            "   uint data[]; } datay;\n"
            "layout(binding = 1) readonly buffer DataU {\n"
            "   uint data[]; } datau;\n"
            "layout(binding = 2) readonly buffer DataV {\n"
            "   uint data[]; } datav;\n"
            "layout(binding = 3)  buffer DataRgba{ \n"
            "   uint data[]; } datargba; \n"
            "uniform uint width; \n"     //这里的宽高都是以u图的尺寸为标准，同时由于一个int表示了4个u，所以图片的实际宽度为width*4*2
            "uniform uint height; \n"    //图片的实际高度为height*2
            " uint four = uint(4); \n"
            " uint two =  uint(2); \n"
            " uint one =  uint(1); \n"
            " \n"
            "void main() { \n"
            "   uint index_uv = gl_GlobalInvocationID.y * width + gl_GlobalInvocationID.x; \n"
            "   vec4 u = unpackUnorm4x8(datau.data[index_uv]); \n"
            "   vec4 v = unpackUnorm4x8(datav.data[index_uv]); \n"
            "   uint index_y1 = gl_GlobalInvocationID.y * width * four + gl_GlobalInvocationID.x * two; \n"
            "   vec4 y1 = unpackUnorm4x8(datay.data[index_y1]); \n"
            "   uint index_y2 = gl_GlobalInvocationID.y * width * four + gl_GlobalInvocationID.x * two + one; \n"
            "   vec4 y2 = unpackUnorm4x8(datay.data[index_y2]); \n"
            "   uint index_y3 = gl_GlobalInvocationID.y * width * four + two * width + gl_GlobalInvocationID.x * two; \n"
            "   vec4 y3 = unpackUnorm4x8(datay.data[index_y3]); \n"
            "   uint index_y4 = gl_GlobalInvocationID.y * width * four + two * width + gl_GlobalInvocationID.x * two + one; \n"
            "   vec4 y4 = unpackUnorm4x8(datay.data[index_y4]); \n"
            "   vec4 rgba[16]; \n"
            "   vec4 ys[4]; \n"
            "   ys[0] =y1; \n"
            "   ys[1] =y2; \n"
            "   ys[2] =y3; \n"
            "   ys[3] =y4; \n"
            "   for(int i = 0; i < 8; ++i){ \n"
            "       rgba[i].x = ys[i / 4][i % 4] + 1.402 * (v[i / 2] - 0.5); \n"
            "       rgba[i].y = ys[i / 4][i % 4] - 0.34414 * (u[i / 2] - 0.5) - 0.71414 * (v[i / 2] - 0.5); \n"
            "       rgba[i].z = ys[i / 4][i % 4] + 1.772 * (u[i / 2] - 0.5); \n"
            "       rgba[i].w = 1.0; \n"    //第一行处理结束
            "       rgba[i+8].x = ys[i / 4+2][i % 4] + 1.402 * (v[i / 2] - 0.5); \n"
            "       rgba[i+8].y = ys[i / 4+2][i % 4] - 0.34414 * (u[i / 2] - 0.5) - 0.71414 * (v[i / 2] - 0.5); \n"
            "       rgba[i+8].z = ys[i / 4+2][i % 4] + 1.772 * (u[i / 2] - 0.5); \n"
            "       rgba[i+8].w = 1.0; \n"      //第二行处理结束
            "   }\n"
            "   for(int i = 0; i < 8; ++i){ \n" //处理第一行
            "       datargba.data[gl_GlobalInvocationID.y* two * width*four*two + gl_GlobalInvocationID.x * four * two + uint(i)] = packUnorm4x8(rgba[i]); \n"
            "       datargba.data[(gl_GlobalInvocationID.y* two + one) * width*four*two +  gl_GlobalInvocationID.x * four * two + uint(i)] = packUnorm4x8(rgba[i+8]); \n"
            "   } \n"
            "}"
    };


    const char srcv[] = {
            "#version 320 es\n"
            "layout(binding = 0) buffer DataRgba{ \n"
            "   uint data[];} datargba; \n"
            "in vec4 pos;\n"
            "uniform int width;\n"
            "uniform int height;\n"
            "in vec2 posrgba; \n"
            "out vec4 color; \n"
            "void main(){\n"
            "   gl_Position = pos;\n"
            "   color = unpackUnorm4x8(datargba.data[uint(posrgba.y * height * width + width * posrgba.x)]);\n"
            "}"
    };
    const char srcf[] = {
            "#version 320 es\n"
            "in vec4 color;\n"
            "out vec4 fcolor;\n"
            "void main(){fcolor = color;}"
    };


    GLuint program = createComputeProgram(src);
    glUseProgram(program);
    CHECK();
    GLint x,y,z,invo;
    glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 0, &x);
    glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 1, &y);
    glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 2, &z);
    glGetIntegerv(GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS,  &invo);
    LOG("max work group %d %d %d %d", x,y,z, invo);
    CHECK();
    //return;
    cv::Mat img = cv::imread("/sdcard/in.jpg", 1);
    cv::resize(img, img, cv::Size(32 * 32, 32 * 32 ));
    cv::Mat yuv;
    cv::cvtColor(img, yuv, CV_BGR2YUV_I420);
    cv::imwrite("/sdcard/yuv.bmp", yuv);
    LOG("%d   yuv size w*h=%d*%d", yuv.isContinuous(), yuv.cols, yuv.rows);
    int width = img.cols;
    int height = img.rows;
    uchar *ptry = (uchar *)yuv.data;
    uchar *ptru = ptry + width * height;
    uchar *ptrv = ptru + width * height / 4;
    width = width / 2 / 4;
    height = height / 2 / 4;
    int length = img.cols * img.rows;
    CHECK();
    GLint locationw = glGetUniformLocation(program, "width");
    //GLint locationh = glGetUniformLocation(program, "height");
    LOG("uniform location %d ", locationw );
    glUniform1ui(locationw, GLuint( width));
    //glUniform1i(locationh, height);
    CHECK();
    GLuint ssbos[4];
    glGenBuffers(4, ssbos);
    LOG("bing data start");
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbos[0]);
    glBufferData(GL_SHADER_STORAGE_BUFFER, length, ptry, GL_STATIC_READ);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbos[0]);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbos[1]);
    glBufferData(GL_SHADER_STORAGE_BUFFER, length / 4, ptru, GL_STATIC_READ);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssbos[1]);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbos[2]);
    glBufferData(GL_SHADER_STORAGE_BUFFER, length / 4, ptrv, GL_STATIC_READ);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, ssbos[2]);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbos[3]);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(uint) * length, NULL, GL_STATIC_READ);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, ssbos[3]);
    LOG("bind end ssbo");
    CHECK();
    glUseProgram(program);
    glDispatchCompute(32 * 32, 32*32 , 1);
    LOG("dispatch compute");
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    CHECK();

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbos[3]);
    LOG("get ptr 0");
    uchar *ptrrgba = (uchar*)glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, sizeof(uint) * length, GL_MAP_READ_BIT);
    LOG("get ptr");
    cv::Mat result(img.rows, img.cols, CV_8UC4, ptrrgba);
    cv::cvtColor(result, result, CV_RGBA2BGR);
    LOG("result r*c:%d %d", result.rows, result.cols);
    cv::imwrite("/sdcard/out.jpg", result);
    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);

    glDeleteProgram(program);
    CHECK();
    LOG("end");


    /*******show result********/
    program = glCreateProgram();
    GLuint vshader = loadShader(GL_VERTEX_SHADER, srcv);
    GLuint fshader = loadShader(GL_FRAGMENT_SHADER, srcf);
    glAttachShader(program, vshader);
    glAttachShader(program, fshader);
    glLinkProgram(program);
    CHECK();

}

void myRgbComputeShader()
{
    struct RGBA{
        unsigned char r;
        unsigned char g;
        unsigned char b;
        unsigned char a;
    };

    const char rgbcs[] ={
            "#version 320 es\n"
            "layout(local_size_x = 10) in;\n"
            "layout(binding = 0) writeonly buffer Output{\n"
            "   uvec4 rgbdata[400];\n"
            "} output0;\n"
            "void main(){\n"
            "   //uvec3 data = uvec3(gl_GlobalInvocationID.x % 256, gl_GlobalInvocationID.y % 256, 128);\n"
            "   output0.rgbdata[gl_GlobalInvocationID.x] = uvec4(1,1,1,1);\n"
            "}"
    };


    GLuint computeProgram;
    GLuint outputSSbo;

    CHECK();
    computeProgram = createComputeProgram(rgbcs);
    LOG("create program end");
    CHECK();


    LOG("data start");
    glGenBuffers(1, &outputSSbo);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, outputSSbo);

    glBufferData(GL_SHADER_STORAGE_BUFFER, 400 * sizeof(RGBA), NULL, GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, outputSSbo);
    LOG("data end");
    CHECK();

    glUseProgram(computeProgram);
    glDispatchCompute(40, 30, 1);   // arraySize/local_size_x

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    CHECK();

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, outputSSbo);
    unsigned char *pOut = (unsigned char *) glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, 400*300*3 * sizeof(unsigned char),
                                             GL_MAP_READ_BIT);
    LOG("result %d %d %d ", pOut[0], pOut[1], pOut[2]);
    LOG("result %d %d %d ", pOut[3], pOut[4], pOut[5]);
    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    LOG("verification PASSED\n");
    glDeleteProgram(computeProgram);
}


void tryComputeShader() {
    //myRgbComputeShader();
    //myImageStoreTest();
    yuvToRgb();
    return;


    GLuint computeProgram;
    GLuint input0SSbo;
    GLuint input1SSbo;
    GLuint outputSSbo;

    CHECK();
    computeProgram = createComputeProgram(gComputeShader);
    LOG("create program end");
    CHECK();

    const GLuint arraySize = 8000;
    unsigned char f0[arraySize];
    unsigned char f1[arraySize];
    for (GLuint i = 0; i < arraySize; ++i) {
        f0[i] = (unsigned char)(i % 100);
        f1[i] = (unsigned char)(i % 100);
    }

    LOG("data start");
    setupSSBufferObject(input0SSbo, 0, f0, arraySize);
    setupSSBufferObject(input1SSbo, 1, f1, arraySize);
    setupSSBufferObject(outputSSbo, 2, NULL, arraySize);
    LOG("data end");
    CHECK();

    glUseProgram(computeProgram);
    glDispatchCompute(1000, 1, 1);   // arraySize/local_size_x
    LOG("dispathc");
    CHECK();
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    CHECK();

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, input0SSbo);
    unsigned char *pOut = (unsigned char *) glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, arraySize * sizeof(unsigned char),
                                             GL_MAP_READ_BIT);
    CHECK();
    LOG("result %x %x %x %x ", pOut[0], pOut[1], pOut[arraySize - 2], pOut[arraySize - 1]);
    for (GLuint i = 0; i < arraySize/4; ++i) {
        if (pOut[i] != (f0[i] + f1[i])) {
            LOG("verification FAILED at array index %d, actual: %d, expected: %d\n", i, pOut[i],
                f0[i] + f1[i]);
            glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
            return;
        }
    }
    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    LOG("verification PASSED\n");
    glDeleteProgram(computeProgram);
}

int main(int /*argc*/, char** /*argv*/)
{
    EGLDisplay dpy = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (dpy == EGL_NO_DISPLAY) {
       LOG("eglGetDisplay returned EGL_NO_DISPLAY.\n");
        return 0;
    }

    EGLint majorVersion;
    EGLint minorVersion;
    EGLBoolean returnValue = eglInitialize(dpy, &majorVersion, &minorVersion);
    LOG("version opengles %d.%d", majorVersion, minorVersion);
    if (returnValue != EGL_TRUE) {
       LOG("eglInitialize failed\n");
        return 0;
    }

    EGLConfig cfg;
    EGLint count;
    EGLint s_configAttribs[] = {
            EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT_KHR,
            EGL_NONE };
    if (eglChooseConfig(dpy, s_configAttribs, &cfg, 1, &count) == EGL_FALSE) {
       LOG("eglChooseConfig failed\n");
        return 0;
    }
    LOG("egl config end");

    EGLint context_attribs[] = { EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE };
    EGLContext context = eglCreateContext(dpy, cfg, EGL_NO_CONTEXT, context_attribs);
    if (context == EGL_NO_CONTEXT) {
       LOG("eglCreateContext failed\n");
        return 0;
    }
    returnValue = eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, context);
    if (returnValue != EGL_TRUE) {
       LOG("eglMakeCurrent failed returned %d\n", returnValue);
        return 0;
    }
    LOG("egl create context");

    tryComputeShader();

    eglDestroyContext(dpy, context);
    eglTerminate(dpy);

    return 0;
}


extern "C" JNIEXPORT jstring

JNICALL
Java_willhua_opengles320test_MainActivity_stringFromJNI(
        JNIEnv *env,
        jobject /* this */) {

    __android_log_print(ANDROID_LOG_DEBUG, "LYH", "jni start");


    main(0,0);
    __android_log_print(ANDROID_LOG_DEBUG, "LYH", "jni main end");

    std::string hello = "Hello from C++";
    return env->NewStringUTF(hello.c_str());
}
