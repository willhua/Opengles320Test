//
// Created by F2845967 on 2018/6/14.
//

#include <EGL/egl.h>
#include <EGL/eglext.h> //仅仅为了EGL_OPENGL_ES3_BIT_KHR
#include <GLES3/gl32.h>
#include <malloc.h>


int esmain()
{
    //获取显示设备，建立与显示设备的链接
    EGLDisplay eglDisplay = eglGetCurrentDisplay();
    EGLint majorversion, minorversion;
    EGLBoolean status = eglInitialize(eglDisplay, &majorversion, &minorversion);//初始化egl
    if (status == EGL_FALSE)
    {
        //fail
    }

    //查询有多少种可以使用的config
    EGLint numconfigs;
    eglGetConfigs(eglDisplay, NULL, 0, &numconfigs);


    //获取所有的可用config信息
    EGLConfig *eglConfigs = (EGLConfig*)malloc(sizeof(EGLConfig) * numconfigs);
    eglGetConfigs(eglDisplay, eglConfigs, numconfigs, &numconfigs);


    //查询第二个配置中的buffer_size的值，将保存在attributevalue中
    EGLint attributevalue;
    eglGetConfigAttrib(eglDisplay, eglConfigs[1], EGL_BUFFER_SIZE, &attributevalue);
    free(eglConfigs);

    //根据属性要求，让系统查询满足条件的配置,返回到configs里
    EGLint  attributeList[]= {
            EGL_RED_SIZE, 5,
            EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT_KHR, //EGL_OPENGL_ES3_BIT_KHR这个值位于eglext.h中
            EGL_GREEN_SIZE, 6,
            EGL_DEPTH_SIZE, 1,
            EGL_NONE
    };
    const EGLint maxconfigs = 5;   //让最多返回5个
    EGLConfig configs[maxconfigs];
    eglChooseConfig(eglDisplay, attributeList, configs, maxconfigs, &numconfigs);
    EGLConfig myeglconfig = configs[0];


    //创建屏幕上的渲染区域：EGL窗口，即绘图场所
    EGLint attribute[] = {
            EGL_RENDER_BUFFER, EGL_BACK_BUFFER,
            EGL_NONE
    };
    EGLNativeWindowType nativeWindow;   //这个要从其他途径获取https://www.gamedev.net/forums/topic/673301-android-native-window-create-destroy-and-opengl-context/
    EGLSurface eglSurface = eglCreateWindowSurface(eglDisplay, myeglconfig, nativeWindow, attribute);


    //创建渲染上下文
    EGLint attribute2[] = {     //也因为我们使用es3.x，而其默认值为1，所以必须要这个属性
            EGL_CONTEXT_CLIENT_VERSION, 3,
            EGL_NONE
    };
    EGLContext eglContext = eglCreateContext(eglDisplay, myeglconfig,
                                             EGL_NO_CONTEXT, //表示没有共享context
                                             attribute2);
    if (eglContext == EGL_NO_CONTEXT)
    {
        //fail
    }


    //指定当前eglcontext
    eglMakeCurrent(eglDisplay, eglSurface, eglSurface, eglContext );    //对于computeshader，就可以不用surface,用EGL_NO_SURFACE即可




}

