#include "cinder/app/App.h"
#include "cinder/app/RendererGl.h"
#include "cinder/gl/gl.h"
#include "cinder/CinderImGui.h"
#include "cinder/Log.h"
#include "cinder/Utilities.h"
#include "cinder/FileWatcher.h"
#include "MotionProcessor.h"
#include "cinder/osc/Osc.h"

using namespace ci;
using namespace ci::app;
using namespace std;
using SenderRef = std::shared_ptr<osc::SenderUdp>;

class MotionTrackerApp : public App {
protected:
    MotionProcessorRef  mMotion;
    ci::gl::GlslProgRef mBlendGlsl;
    SenderRef           mSkeletonSender;
    
    uint16_t    mDestinationPort, mLocalPort;
    std::string mRemoteIp;
    
    Rectf mBound;
    
    float mTime;
    bool  mDrawUi;
    
public:
    void setup() override;
    void mouseDown( MouseEvent event ) override;
    void keyDown( KeyEvent event ) override;
    void update() override;
    void draw() override;

    void onSendError( asio::error_code error );
    void saveSettings();
    void loadSettings();
    void drawUi();
};

void MotionTrackerApp::setup() {
    app::addAssetDirectory(app::getAppPath());
    
    // enable this if you need files to be logged
    //log::makeLogger<ci::log::LoggerFileRotating>("/Users/USER/Documents/SolidJellyfishLtd/logging", "Tracker.%Y.%m.%d.log");
    
    auto option     = ImGui::Options();
    ImGuiStyle style= ImGuiStyle();
    style.ScaleAllSizes (app::toPixels(.75f));
    option.style        (style);
    ImGui::Initialize   (option);
    ImGuiIO& io = ImGui::GetIO();
    auto fontData = app::loadResource(UI_FONT);
    io.Fonts->AddFontFromFileTTF(fontData->getFilePathHint().string().c_str(), app::toPixels(12));
    
    // if using more camera than 2, can potentially make the main thread run faster
    gl::enableVerticalSync(false);
    app::setFrameRate(90.f);
    mDrawUi = true;
    
    {
        try{
            mBlendGlsl = gl::GlslProg::create(app::loadResource(BLEND_VERT),
                                              app::loadResource(BLEND_FRAG));
        } catch(const ci::Exception& ex) {
            CI_LOG_EXCEPTION("Blend glsl", ex);
        }
#if SHADER_RELOAD
        auto path = {app::getAssetPath("shaders/blend.vert"),
            app::getAssetPath("shaders/blend.frag")};
        FileWatcher::instance().watch(path, [this](const WatchEvent &){
            try{
                auto glsl = gl::GlslProg::create(app::loadAsset("shaders/blend.vert"),
                                         app::loadAsset("shaders/blend.frag"));
                mBlendGlsl = glsl;
            } catch(const ci::Exception& ex) {
                CI_LOG_EXCEPTION("Blend glsl", ex);
            }
        });
#endif
    }
    
    loadSettings();
    
    // osc
    {
        mSkeletonSender  = make_shared<osc::SenderUdp>(mLocalPort, mRemoteIp, mDestinationPort);
        try {
            mSkeletonSender->bind();
        } catch ( const osc::Exception &ex ) {
            CI_LOG_E( "Error binding: " << ex.what() << " val: " << ex.value() );
            quit();
        }
    }
    
    mTime   = (float)app::getElapsedSeconds();
}

void MotionTrackerApp::loadSettings() {
    try {
        nlohmann::json file = nlohmann::json::parse(loadString(app::loadAsset("settings.json")));
        mMotion = MotionProcessor::create(file["motion"]);
        
        {
            auto& osc        = file["osc"];
            mDestinationPort = osc["port"];
            mLocalPort       = 10000;
            mRemoteIp        = osc["ip"];// "127.0.0.1";
        }
        
        // bound
        {
            auto& bnd = file["bound"];
            mBound.set(bnd["x1"], bnd["y1"], bnd["x2"], bnd["y2"]);
        }
        
        FileWatcher::instance().watch("settings.json", [this](const WatchEvent& evt){
            try {
                nlohmann::json file = nlohmann::json::parse(loadString(app::loadAsset("settings.json")));
                mMotion->load(file["motion"]);
            } catch(const ci::Exception& ex) {
                CI_LOG_EXCEPTION("Settings file update", ex);
            }
        });
    } catch(const Exception& ex) {
        CI_LOG_EXCEPTION("Settings file load err: ", ex);
        
        // init with default values
        mMotion          = MotionProcessor::create();
        mDestinationPort = 8080;
        mLocalPort       = 10000;
        mRemoteIp        = "127.0.0.1";
    }
}

void MotionTrackerApp::saveSettings(){
    nlohmann::json saveFile = nlohmann::json::object();
    
    saveFile["motion"] = mMotion->toJson();

    {
        nlohmann::json osc = nlohmann::json::object();
        osc["port"]        = mDestinationPort;
        osc["localport"]   = mLocalPort;
        osc["ip"]          = mRemoteIp;
        saveFile["osc"]    = osc;
    }
    
    // bound
    {
        nlohmann::json bnd = nlohmann::json::object();
        bnd["x1"] = mBound.x1;
        bnd["x2"] = mBound.x2;
        bnd["y1"] = mBound.y1;
        bnd["y2"] = mBound.y2;
        saveFile["bound"] = bnd;
    }
    
    string filepath  = app::getAssetPath("") / "settings.json";
    std::ofstream ostream(filepath);
    std::string data = saveFile.dump(4);
    ostream << data << endl;
}

void MotionTrackerApp::onSendError( asio::error_code error ) {
    if( error ) {
        CI_LOG_E( "Error sending: " << error.message() << " val: " << error.value() );
        /*try {
            mSkeletonSender->close();
        } catch( const osc::Exception &ex ) {
            CI_LOG_EXCEPTION( "Cleaning up socket: val -" << ex.value(), ex );
        }*/
    }
}

void MotionTrackerApp::mouseDown( MouseEvent event ) {
    if(event.isRight())
        mMotion->addDepthPoint(vec2(event.getPos()) /
                               vec2(app::getWindowSize()));
}

void MotionTrackerApp::keyDown( KeyEvent event ) {
    if(event.getCode() == KeyEvent::KEY_ESCAPE)
        quit();
    else if(event.getCode() == KeyEvent::KEY_u)
        mDrawUi = !mDrawUi;
    else if(event.isMetaDown() && event.getCode() == KeyEvent::KEY_s)
        saveSettings();
}

void MotionTrackerApp::update() {
    {
        float ct = (float)app::getElapsedSeconds();
        float dt = glm::clamp(ct - mTime, 1.f/120.f, 1.f/15.f);
        
        // process camera feed and ml data
        if(mMotion->update(dt, mTime)){
            auto num = mMotion->availableSkeletonCnt();
            osc::Message msg( "/skeleton" );
            if(num > 0) {
                // idx, x, z, angle
                vector<vec4> data;
                int32_t filteredCnt = 0;
                for(size_t i = 0; i < num; i++ ) {
                    auto bodyData = mMotion->skeletonData(i);
                    vec3& waist   = bodyData[7];
                    vec3& neck    = bodyData[8];
                    
                    // average neck and waist to get chest
                    vec3 piv      = (waist + neck) / 2.f;
                    
                    // if body is active and within boundary
                    if(bodyData[0].z > .5f && mBound.contains(vec2(piv.x,piv.y))) {
                        data.emplace_back(vec4( // body idx
                                                bodyData[0].x,
                                                // normalized position in boundary
                                                (piv.x - mBound.getX1()) / mBound.getWidth(),
                                                (piv.y - mBound.getY1()) / mBound.getHeight(),
                                                0.f ));
                        
                        // add other custom data you need
                        
                        filteredCnt ++;
                    }
                }
                
                msg.append((int32_t)filteredCnt);
                if(!data.empty())
                    msg.appendBlob(data.data(), sizeof(vec4) * (uint32_t)data.size());
            } else msg.append((int32_t)0);
            
            mSkeletonSender->send( msg,
            std::bind( &MotionTrackerApp::onSendError, this, std::placeholders::_1 ) );
        }
        
        mTime = ct;
    }
    
    drawUi();
}

void MotionTrackerApp::drawUi(){
    if(!mDrawUi) return;
    
    ImGui::ScopedWindow scpWin("Params");
    ImGui::SetWindowPos (app::toPixels(vec2(10, 10)));
    ImGui::SetWindowSize(app::toPixels(vec2(340,600)));
    ImGui::Text("Active Bound");
    ImGui::InputFloat("X1", &mBound.x1); ImGui::InputFloat("X2", &mBound.x2);
    ImGui::InputFloat("Y1", &mBound.y1); ImGui::InputFloat("Y2", &mBound.y2);
    ImGui::Dummy(ivec2(0,4));
    mMotion->drawUi();
}

void MotionTrackerApp::draw() {
    gl::clear( Color( 0, 0, 0 ) );
    mMotion->draw(mBound);
}

CINDER_APP( MotionTrackerApp, RendererGl, [](App::Settings *settings){
    settings->setHighDensityDisplayEnabled(true);
    settings->setWindowSize (1280, 720);
    settings->setResizable  (false);
} )
