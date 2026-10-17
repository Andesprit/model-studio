plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.andesprit.detectors"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.andesprit.detectors"
        minSdk = 28 // MediaMetadataRetriever.getFrameAtIndex
        targetSdk = 35
        versionCode = 1
        versionName = "0.1"
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        // phones + the emulator; -Pabi=arm64-v8a builds a phone-only APK about half the size
        ndk { abiFilters += (findProperty("abi") as String?)?.split(",") ?: listOf("arm64-v8a", "x86_64") }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            signingConfig = signingConfigs.getByName("debug") // sideloaded, never published
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
    androidResources { noCompress += listOf("tflite", "onnx") } // models are memory-mapped from the APK
}

dependencies {
    implementation("androidx.activity:activity-ktx:1.9.3")
    implementation("androidx.camera:camera-camera2:1.4.0")
    implementation("androidx.camera:camera-lifecycle:1.4.0")
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.20.0")
    implementation("com.google.ai.edge.litert:litert:1.0.1")
    implementation("com.google.ai.edge.litert:litert-gpu:1.0.1")
    implementation("org.opencv:opencv:4.10.0")

    androidTestImplementation("androidx.test.ext:junit:1.2.1")
    androidTestImplementation("androidx.test:runner:1.6.2")
}
