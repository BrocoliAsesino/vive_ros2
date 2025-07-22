#include <string>
#include <openvr.h>
#include <chrono> // Include this for std::chrono
#include <thread>

#include "VRUtils.hpp"
#include "json.hpp" // Include nlohmann/json
#include "server.hpp"


// Forward declaration for VRTrackerData
// This struct will hold the pose and velocity data for a generic tracker
struct VRTrackerData {
    long long time;
    uint32_t device_index; // Use uint32_t for device index as per OpenVR
    Eigen::Vector3d position;
    Eigen::Quaterniond quaternion;
    Eigen::Vector3d linear_velocity;
    Eigen::Vector3d angular_velocity;
    bool pose_is_valid;
    bool device_is_connected;

    // Helper to reset data
    void reset() {
        time = 0;
        device_index = vr::k_unTrackedDeviceIndexInvalid; // Use OpenVR's invalid index
        position.setZero();
        quaternion.setIdentity();
        linear_velocity.setZero();
        angular_velocity.setZero();
        pose_is_valid = false;
        device_is_connected = false;
    }
};
// VR Input Configuration Constants
namespace VRInputConfig {
    constexpr double DEFAULT_PUBLISH_FREQUENCY = 50.0;
    constexpr double MIN_PUBLISH_FREQUENCY = 0.1;
    constexpr double MAX_PUBLISH_FREQUENCY = 1000.0;
    constexpr int TRIGGER_FEEDBACK_STEPS = 6;
    constexpr int HAPTIC_FEEDBACK_DURATION = 200;
    constexpr int TRIGGER_HAPTIC_MULTIPLIER = 3000;
    constexpr int WARNING_HAPTIC_DURATION = 20;
    constexpr float POSITION_THRESHOLD = 0.05f;
    constexpr float DISTANCE_THRESHOLD = 0.1f;
    constexpr int NO_CONTROLLER_SLEEP_MS = 50;
    constexpr int DATA_COLLECTION_SLEEP_MS = 5;
    constexpr int CONTROLLER_DELAY_MS = 1;
    constexpr int LOG_INTERVAL_SECONDS = 1;
    constexpr double Y_OFFSET = 0.6;
    
    // Controller indices
    constexpr int RIGHT_CONTROLLER_INDEX = 0;
    constexpr int LEFT_CONTROLLER_INDEX = 1;
    constexpr int MAX_CONTROLLERS = 2;
    constexpr int ROLE_TRACKER0 = 2;
}

// VR Controller Input Handler - manages VR device detection and data processing
// VR Controller Input Handler - manages VR device detection and data processing
class ViveInput {
public:
    ViveInput(std::mutex &mutex, std::condition_variable &cv, VRControllerData &data, double publish_freq = VRInputConfig::DEFAULT_PUBLISH_FREQUENCY);
    ~ViveInput();
    void runVR();
    void setPublishFrequency(double freq);

private:
    vr::IVRSystem *pHMD = nullptr;
    vr::EVRInitError eError = vr::VRInitError_None;
    vr::TrackedDevicePose_t trackedDevicePose[vr::k_unMaxTrackedDeviceCount];

    std::mutex &data_mutex;
    std::condition_variable &data_cv;
    VRControllerData &shared_data;
    VRControllerData local_controller_data;
    
    // Track both controllers separately (right = 0, left = 1)
    bool controller_detected[VRInputConfig::MAX_CONTROLLERS] = {false, false};
    Eigen::Vector3d prev_position[VRInputConfig::MAX_CONTROLLERS];  // Use Eigen for better vector operations
    std::chrono::steady_clock::time_point prev_time[VRInputConfig::MAX_CONTROLLERS];
    bool first_run[VRInputConfig::MAX_CONTROLLERS] = {true, true};
    
    // Store data for both controllers separately
    VRControllerData right_controller_data; // For right controller (role_index = 0)
    VRControllerData left_controller_data;  // For left controller (role_index = 1)
    bool right_controller_updated = false;
    bool left_controller_updated = false;
    
    // Configurable publishing frequency for each controller (in Hz)
    double controllerPublishFrequency = VRInputConfig::DEFAULT_PUBLISH_FREQUENCY;
    std::chrono::steady_clock::time_point last_publish_time[VRInputConfig::MAX_CONTROLLERS]; // For each controller
    bool publish_time_initialized[VRInputConfig::MAX_CONTROLLERS] = {false, false};

    // --- New members for HTC Vive Trackers ---
    std::vector<VRTrackerData> tracker_data;
    std::vector<bool> tracker_detected;
    std::vector<Eigen::Vector3d> prev_tracker_position;
    std::vector<std::chrono::steady_clock::time_point> prev_tracker_time;
    std::vector<bool> first_tracker_run;
    std::vector<bool> tracker_updated;
    std::vector<std::chrono::steady_clock::time_point> last_tracker_publish_time;
    std::vector<bool> tracker_publish_time_initialized;
    // --- End new members ---

    bool initVR();
    bool shutdownVR();
    void processControllerData(uint32_t deviceIndex, int roleIndex);
    void processControllerButtons(uint32_t deviceIndex, vr::VRControllerState_t& controllerState, int roleIndex);
    void processTriggerFeedback(float triggerValue, uint32_t deviceIndex, int roleIndex);
    // bool validatePositionChange(const Eigen::Vector3d& currentPosition, int roleIndex);
    // Generalized position validation method
    bool validatePositionChange(const Eigen::Vector3d& currentPosition, uint32_t deviceIndex, bool isController);
    // New method for processing tracker data
    void processTrackerData(uint32_t deviceIndex);

    void publishControllerDataAtFrequency();
    void handleControllerDetection(bool anyControllerDetected, std::chrono::steady_clock::time_point currentTime, std::chrono::steady_clock::time_point& lastLogTime);
};

ViveInput::ViveInput(std::mutex &mutex, std::condition_variable &cv, VRControllerData &data, double publish_freq) 
    : data_mutex(mutex), data_cv(cv), shared_data(data), controllerPublishFrequency(publish_freq) {
    if (!initVR()) {
        shutdownVR();
        throw std::runtime_error("Failed to initialize VR");
    }

    // Initialize tracker-specific vectors to max possible device count
    tracker_data.resize(vr::k_unMaxTrackedDeviceCount);
    tracker_detected.resize(vr::k_unMaxTrackedDeviceCount, false);
    prev_tracker_position.resize(vr::k_unMaxTrackedDeviceCount);
    prev_tracker_time.resize(vr::k_unMaxTrackedDeviceCount);
    first_tracker_run.resize(vr::k_unMaxTrackedDeviceCount, true);
    tracker_updated.resize(vr::k_unMaxTrackedDeviceCount, false);
    last_tracker_publish_time.resize(vr::k_unMaxTrackedDeviceCount);
    tracker_publish_time_initialized.resize(vr::k_unMaxTrackedDeviceCount, false);

    // Reset initial states for all potential trackers
    for (uint32_t i = 0; i < vr::k_unMaxTrackedDeviceCount; ++i) {
        tracker_data[i].reset();
        prev_tracker_position[i].setZero(); // Initialize Eigen vectors
    }
}

ViveInput::~ViveInput() {
    shutdownVR();
}

void ViveInput::setPublishFrequency(double freq) {
    if (freq > VRInputConfig::MIN_PUBLISH_FREQUENCY && freq <= VRInputConfig::MAX_PUBLISH_FREQUENCY) {
        controllerPublishFrequency = freq;
        logMessage(Info, "Controller publish frequency updated to: " + std::to_string(freq) + " Hz");
        
        // Reset timing for both controllers to apply new frequency immediately
        publish_time_initialized[VRInputConfig::RIGHT_CONTROLLER_INDEX] = false;
        publish_time_initialized[VRInputConfig::LEFT_CONTROLLER_INDEX] = false;

        // Reset timing for all trackers as well
        for (uint32_t i = 0; i < vr::k_unMaxTrackedDeviceCount; ++i) {
            tracker_publish_time_initialized[i] = false;
        }
    } else {
        logMessage(Warning, "Invalid frequency: " + std::to_string(freq) + " Hz. Must be between " + 
                   std::to_string(VRInputConfig::MIN_PUBLISH_FREQUENCY) + " and " + 
                   std::to_string(VRInputConfig::MAX_PUBLISH_FREQUENCY) + " Hz");
    }
}

void ViveInput::runVR() {
    logMessage(Info, "Starting VR loop");
    logMessage(Info, "Controller publish frequency set to: " + std::to_string(controllerPublishFrequency) + " Hz");
    auto lastLogTime = std::chrono::steady_clock::now(); // Initialize the last log time
    
    // For trigger feedback
    static int previousStep[VRInputConfig::MAX_CONTROLLERS] = {-1, -1}; // Initialize previous step for both controllers

    while (true) {
        // Reset controller detection status for this loop
        controller_detected[0] = false; // Right controller
        controller_detected[1] = false; // Left controller

        for (uint32_t i = 0; i < vr::k_unMaxTrackedDeviceCount; ++i) {
            tracker_detected[i] = false;
            tracker_updated[i] = false; // Reset tracker update flags
        }
        
        // Reset controller update flags for this iteration
        right_controller_updated = false;
        left_controller_updated = false;
        
        // Update the poses for all devices
        pHMD->GetDeviceToAbsoluteTrackingPose(vr::TrackingUniverseStanding, 0, trackedDevicePose, vr::k_unMaxTrackedDeviceCount);

        // First, scan for all controllers
        for (uint32_t i = 0; i < vr::k_unMaxTrackedDeviceCount; i++) {
            if (trackedDevicePose[i].bDeviceIsConnected && trackedDevicePose[i].bPoseIsValid
                && trackedDevicePose[i].eTrackingResult == vr::TrackingResult_Running_OK) {

                vr::ETrackedDeviceClass trackedDeviceClass = pHMD->GetTrackedDeviceClass(i); 


                // if (VRUtils::controllerIsConnected(pHMD, i)) {
                //     // Get the controller's role (left or right)
                //     vr::ETrackedControllerRole controllerRole = VRUtils::controllerRoleCheck(pHMD, i);
                if (trackedDeviceClass == vr::ETrackedDeviceClass::TrackedDeviceClass_Controller) {
                    // This is a controller (e.g., Vive Wand)
                    // Get the controller's role (left or right)
                    vr::ETrackedControllerRole controllerRole = pHMD->GetControllerRoleForTrackedDeviceIndex(i);

                    // Process only if it's a valid controller (left or right hand)
                    if (controllerRole == vr::TrackedControllerRole_LeftHand || 
                        controllerRole == vr::TrackedControllerRole_RightHand) {
                        
                        // Map the controller role to our index (0=right, 1=left)
                        int role_index = (controllerRole == vr::TrackedControllerRole_LeftHand) ? 1 : 0;
                        
                        // Mark this controller as detected
                        controller_detected[role_index] = true;
                        
                        // Process this controller's data using Eigen transforms
                        // Get the pose of the device
                        vr::HmdMatrix34_t steamVRMatrix = trackedDevicePose[i].mDeviceToAbsoluteTracking;
                        
                        // Use new Eigen-based transform utilities
                        Eigen::Vector3d position = VRTransforms::getPositionFromVRMatrix(steamVRMatrix);
                        Eigen::Quaterniond quaternion = VRTransforms::getQuaternionFromVRMatrix(steamVRMatrix);
                        
                        logMessage(Debug, "[CONTROLLER " + std::to_string(role_index) + "] [POSE CM]: " + 
                                std::to_string(position.x() * 100) + " " + 
                                std::to_string(position.y() * 100) + " " + 
                                std::to_string(position.z() * 100));
                        
                        // Reset local data for this controller
                        VRUtils::resetJsonData(local_controller_data);
                        
                        // Store controller data using new Eigen-based methods
                        local_controller_data.time = Server::getCurrentTimeWithMilliseconds();
                        local_controller_data.role = role_index; // 0 = right, 1 = left
                        local_controller_data.setPosition(position - Eigen::Vector3d(0, VRInputConfig::Y_OFFSET, 0)); // Apply Y offset
                        local_controller_data.setQuaternion(quaternion);

                        // Process controller state and buttons
                        vr::VRControllerState_t controllerState;
                        pHMD->GetControllerState(i, &controllerState, sizeof(controllerState));
                        
                        if ((1LL << vr::k_EButton_ApplicationMenu) & controllerState.ulButtonPressed) {
                            logMessage(Debug, "Application Menu button pressed, resetting the pose");
                            first_run[role_index] = true; // Reset the first run flag for this controller
                            local_controller_data.menu_button = true;
                            VRUtils::HapticFeedback(pHMD, i, 200);
                        }
                        
                        if ((1LL << vr::k_EButton_SteamVR_Trigger) & controllerState.ulButtonPressed) {
                            logMessage(Debug, "Trigger button pressed");
                            local_controller_data.trigger_button = true;
                        }
                        
                        if ((1LL << vr::k_EButton_SteamVR_Touchpad) & controllerState.ulButtonPressed) {
                            logMessage(Debug, "Touchpad button pressed");
                            local_controller_data.trackpad_button = true;
                            VRUtils::HapticFeedback(pHMD, i, 200);
                        }
                        
                        if ((1LL << vr::k_EButton_Grip) & controllerState.ulButtonPressed) {
                            logMessage(Debug, "Grip button pressed");
                            local_controller_data.grip_button = true;
                        }
                        
                        if ((1LL << vr::k_EButton_SteamVR_Touchpad) & controllerState.ulButtonTouched) {
                            logMessage(Debug, "Touchpad button touched");
                            local_controller_data.trackpad_x = controllerState.rAxis[0].x;
                            local_controller_data.trackpad_y = controllerState.rAxis[0].y;
                            local_controller_data.trackpad_touch = true;
                        }
                        
                        // Process trigger
                        const int numSteps = 6;
                        const float stepSize = 1.0f / numSteps;
                        float triggerValue = controllerState.rAxis[1].x;
                        int currentStep = static_cast<int>(triggerValue / stepSize);
                        logMessage(Debug, "Trigger: " + std::to_string(triggerValue) + "\n");
                        local_controller_data.trigger = triggerValue;
                        
                        if (currentStep != previousStep[role_index]) {
                            int vibrationDuration = static_cast<int>(triggerValue * 3000);
                            VRUtils::HapticFeedback(pHMD, i, vibrationDuration);
                            previousStep[role_index] = currentStep;
                        }

                        // Check if the input data is reasonable using Eigen
                        auto current_time = std::chrono::steady_clock::now();
                        if (!first_run[role_index]) {
                            std::chrono::duration<float> time_diff = current_time - prev_time[role_index];
                            float delta_time = time_diff.count();
                            
                            // Calculate position change using Eigen
                            Eigen::Vector3d position_change = position - prev_position[role_index];
                            double delta_distance = position_change.norm();
                            double velocity = delta_distance / delta_time;

                            logMessage(Debug, "[CONTROLLER " + std::to_string(role_index) + "] Velocity: " + 
                                      std::to_string(velocity) + " units/s");
                            logMessage(Debug, "[CONTROLLER " + std::to_string(role_index) + "] Delta pos: " + 
                                      std::to_string(delta_distance) + " units");
                            
                            // Check if delta distance is reasonable using Eigen-based validation
                            if (!VRTransforms::isPositionChangeReasonable(position, prev_position[role_index], 0.05)) {
                                logMessage(Warning, "[CONTROLLER " + std::to_string(role_index) + 
                                          "] Unreasonable delta_distance detected: " + 
                                          std::to_string(delta_distance) + " units. Skipping this data.");
                                VRUtils::HapticFeedback(pHMD, i, VRInputConfig::WARNING_HAPTIC_DURATION);
                                continue; // Skip this data
                            } else {
                                logMessage(Debug, "[CONTROLLER " + std::to_string(role_index) + "] Will publish this data");
                            }
                        } else {
                            first_run[role_index] = false; // Set the flag to false after the first run
                        }

                        // Update previous record
                        prev_position[role_index] = position;
                        prev_time[role_index] = current_time;

                        // Store this controller's data - we'll send both controllers at the end of the loop
                        // Create a copy of the current controller data
                        if (role_index == 0) { // Right controller
                            right_controller_data = local_controller_data;
                            right_controller_updated = true;
                        } else { // Left controller
                            left_controller_data = local_controller_data;
                            left_controller_updated = true;
                        }
                    }
                    else if (trackedDeviceClass == vr::ETrackedDeviceClass::TrackedDeviceClass_GenericTracker) {
                        // This is a generic tracker (e.g., HTC Vive Tracker 3.0)
                        processTrackerData(i); // Process tracker data
                        tracker_detected[i] = true; // Mark this tracker as detected
                    }
                }
            }
        }

        // Process and send controller data at controlled frequency
        // This ensures both controllers are sent at the same configurable rate
        auto currentTime = std::chrono::steady_clock::now();
        bool anyControllerDetected = controller_detected[0] || controller_detected[1];
        bool anyTrackerDetected = false;
        for (uint32_t i = 0; i < vr::k_unMaxTrackedDeviceCount; ++i) {
            if (tracker_detected[i]) {
                anyTrackerDetected = true;
                break;
            }
        }
        
        // Calculate time interval for desired frequency
        double interval_ms = 1000.0 / controllerPublishFrequency;
        auto interval_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::duration<double, std::milli>(interval_ms));
        
        // Check and send data for each controller at controlled frequency
        for (int role = 0; role < 2; role++) {
            bool controller_has_data = (role == 0) ? right_controller_updated : left_controller_updated;
            
            if (controller_has_data) {
                // Initialize publish time if not done yet
                if (!publish_time_initialized[role]) {
                    last_publish_time[role] = currentTime;
                    publish_time_initialized[role] = true;
                }
                
                // Check if enough time has passed for this controller
                auto time_since_last_publish = currentTime - last_publish_time[role];
                if (time_since_last_publish >= interval_duration) {
                    // Send data for this controller
                    {
                        std::lock_guard<std::mutex> lock(data_mutex);
                        if (role == 0) {
                            shared_data = right_controller_data;
                        } else {
                            shared_data = left_controller_data;
                        }
                        shared_data.time = Server::getCurrentTimeWithMilliseconds();
                        shared_data.role = role;
                        data_cv.notify_one();
                    }
                    
                    // Update last publish time for this controller
                    last_publish_time[role] = currentTime;
                    
                    // Log publishing activity
                    std::string controller_name = (role == 0) ? "RIGHT" : "LEFT";
                    logMessage(Debug, "[" + controller_name + "] Data sent to server at " + 
                              std::to_string(controllerPublishFrequency) + " Hz");
                    
                    // Small delay to prevent mutex contention between controllers
                    std::this_thread::sleep_for(std::chrono::milliseconds(VRInputConfig::CONTROLLER_DELAY_MS));
                }
            }
        }

        // --- New: Process and send tracker data at controlled frequency ---
        // Note: The current `shared_data` (VRControllerData) and `Server` class are designed for controllers.
        // To publish tracker data to the ROS2 client, you will need to:
        // 1. Extend `VRControllerData` to be a more generic `VRDeviceData` (e.g., using a `std::variant` or `enum` to differentiate device types).
        // 2. Modify the `Server` class to handle and serialize `VRTrackerData` (or the generalized `VRDeviceData`).
        // 3. Update the Python client (`vive_node`) to deserialize and publish `VRTrackerData` to new ROS2 topics (e.g., `/vive_tracker_X/pose`).
        // For now, the tracker data is processed and stored internally in `tracker_data` vector.
        for (uint32_t i = 0; i < vr::k_unMaxTrackedDeviceCount; ++i) {
            if (tracker_updated[i]) {
                // Initialize publish time if not done yet
                if (!tracker_publish_time_initialized[i]) {
                    last_tracker_publish_time[i] = currentTime;
                    tracker_publish_time_initialized[i] = true;
                }
                
                // Check if enough time has passed for this tracker
                auto time_since_last_publish = currentTime - last_tracker_publish_time[i];
                if (time_since_last_publish >= interval_duration) {
                    // In a real scenario, you would send `tracker_data[i]` to the server here.
                    // For demonstration, we'll just log it.
                    logMessage(Info, " Tracker data ready for publishing. "
                               "Position: (" + std::to_string(tracker_data[i].position.x()) + ", " +
                               std::to_string(tracker_data[i].position.y()) + ", " +
                               std::to_string(tracker_data[i].position.z()) + ")");
                    
                    // Update last publish time for this tracker
                    last_tracker_publish_time[i] = currentTime;
                }
            }
        }
        // --- End new tracker publishing section ---
        
        // Reset controller update flags for the next iteration
        right_controller_updated = false;
        left_controller_updated = false;
        
        // Handle controller detection status
        if (!anyControllerDetected &&!anyTrackerDetected) {
            if (std::chrono::duration_cast<std::chrono::seconds>(currentTime - lastLogTime).count() >= VRInputConfig::LOG_INTERVAL_SECONDS) {
                logMessage(Info, "No controllers or trackers detected, currentTime: " + 
                        std::to_string(std::chrono::duration_cast<std::chrono::seconds>(currentTime.time_since_epoch()).count()));
                
                // Reset first_run flags for both controllers
                first_run[VRInputConfig::RIGHT_CONTROLLER_INDEX] = true;
                first_run[VRInputConfig::LEFT_CONTROLLER_INDEX] = true;

                for (uint32_t i = 0; i < vr::k_unMaxTrackedDeviceCount; ++i) {
                    first_tracker_run[i] = true;
                }
                
                lastLogTime = currentTime; // Update the last log time
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(VRInputConfig::NO_CONTROLLER_SLEEP_MS)); // ~20Hz when no controllers
        } else {
            // Run at higher frequency to ensure responsive data collection
            // but publishing is controlled by controllerPublishFrequency
            std::this_thread::sleep_for(std::chrono::milliseconds(VRInputConfig::DATA_COLLECTION_SLEEP_MS)); // ~200Hz data collection
            lastLogTime = currentTime;
            
            // Log which controllers were detected
            if (controller_detected[VRInputConfig::RIGHT_CONTROLLER_INDEX] && controller_detected[VRInputConfig::LEFT_CONTROLLER_INDEX]) {
                logMessage(Debug, "Both controllers detected");
            } else if (controller_detected[VRInputConfig::RIGHT_CONTROLLER_INDEX]) {
                logMessage(Debug, "Right controller detected");
            } else if (controller_detected[VRInputConfig::LEFT_CONTROLLER_INDEX]) {
                logMessage(Debug, "Left controller detected");
            }
            if (anyTrackerDetected) {
                logMessage(Debug, "Trackers detected.");
            }
        }
    }
}

bool ViveInput::initVR() {
    // Initialize VR runtime
    eError = vr::VRInitError_None;
    pHMD = vr::VR_Init(&eError, vr::VRApplication_Background);
    if (eError != vr::VRInitError_None) {
        pHMD = NULL;
        std::string error_msg = vr::VR_GetVRInitErrorAsEnglishDescription(eError);
        logMessage(Error, "Unable to init VR runtime: " + error_msg);
        return false;
    } else {
        logMessage(Info, "VR runtime initialized");
    }
    return true;
}

bool ViveInput::shutdownVR() {
    // Shutdown VR runtime
    if (pHMD) {
        logMessage(Info, "Shutting down VR runtime");
        vr::VR_Shutdown();
    }
    return true;
}

void ViveInput::processControllerButtons(uint32_t deviceIndex, vr::VRControllerState_t& controllerState, int roleIndex) {
    if ((1LL << vr::k_EButton_ApplicationMenu) & controllerState.ulButtonPressed) {
        logMessage(Debug, "Application Menu button pressed, resetting the pose");
        first_run[roleIndex] = true; // Reset the first run flag for this controller
        local_controller_data.menu_button = true;
        VRUtils::HapticFeedback(pHMD, deviceIndex, VRInputConfig::HAPTIC_FEEDBACK_DURATION);
    }
    
    if ((1LL << vr::k_EButton_SteamVR_Trigger) & controllerState.ulButtonPressed) {
        logMessage(Debug, "Trigger button pressed");
        local_controller_data.trigger_button = true;
    }
    
    if ((1LL << vr::k_EButton_SteamVR_Touchpad) & controllerState.ulButtonPressed) {
        logMessage(Debug, "Touchpad button pressed");
        local_controller_data.trackpad_button = true;
        VRUtils::HapticFeedback(pHMD, deviceIndex, VRInputConfig::HAPTIC_FEEDBACK_DURATION);
    }
    
    if ((1LL << vr::k_EButton_Grip) & controllerState.ulButtonPressed) {
        logMessage(Debug, "Grip button pressed");
        local_controller_data.grip_button = true;
    }
    
    if ((1LL << vr::k_EButton_SteamVR_Touchpad) & controllerState.ulButtonTouched) {
        logMessage(Debug, "Touchpad button touched");
        local_controller_data.trackpad_x = controllerState.rAxis[0].x;
        local_controller_data.trackpad_y = controllerState.rAxis[0].y;
        local_controller_data.trackpad_touch = true;
    }
}

void ViveInput::processTriggerFeedback(float triggerValue, uint32_t deviceIndex, int roleIndex) {
    static int previousStep[VRInputConfig::MAX_CONTROLLERS] = {-1, -1};
    
    const float stepSize = 1.0f / VRInputConfig::TRIGGER_FEEDBACK_STEPS;
    int currentStep = static_cast<int>(triggerValue / stepSize);
    logMessage(Debug, "Trigger: " + std::to_string(triggerValue) + "\n");
    local_controller_data.trigger = triggerValue;
    
    if (currentStep != previousStep[roleIndex]) {
        int vibrationDuration = static_cast<int>(triggerValue * VRInputConfig::TRIGGER_HAPTIC_MULTIPLIER);
        VRUtils::HapticFeedback(pHMD, deviceIndex, vibrationDuration);
        previousStep[roleIndex] = currentStep;
    }
}

// New method to process data for generic trackers
void ViveInput::processTrackerData(uint32_t deviceIndex) {
    // Reset data for this specific tracker
    tracker_data[deviceIndex].reset();

    // Store tracker data
    tracker_data[deviceIndex].time = std::stoll(Server::getCurrentTimeWithMilliseconds()); // Convert string to long long
    tracker_data[deviceIndex].device_index = deviceIndex;

    // Get the pose of the device
    vr::HmdMatrix34_t steamVRMatrix = trackedDevicePose[deviceIndex].mDeviceToAbsoluteTracking;
    Eigen::Vector3d position = VRTransforms::getPositionFromVRMatrix(steamVRMatrix);
    Eigen::Quaterniond quaternion = VRTransforms::getQuaternionFromVRMatrix(steamVRMatrix);

    tracker_data[deviceIndex].position = position;
    tracker_data[deviceIndex].quaternion = quaternion;

    tracker_data[deviceIndex].linear_velocity = Eigen::Vector3d(
        trackedDevicePose[deviceIndex].vVelocity.v[0],
        trackedDevicePose[deviceIndex].vVelocity.v[1],
        trackedDevicePose[deviceIndex].vVelocity.v[2]
    );
    tracker_data[deviceIndex].angular_velocity = Eigen::Vector3d(
        trackedDevicePose[deviceIndex].vAngularVelocity.v[0],
        trackedDevicePose[deviceIndex].vAngularVelocity.v[1],
        trackedDevicePose[deviceIndex].vAngularVelocity.v[2]
    );

    tracker_data[deviceIndex].pose_is_valid = trackedDevicePose[deviceIndex].bPoseIsValid;
    tracker_data[deviceIndex].device_is_connected = trackedDevicePose[deviceIndex].bDeviceIsConnected;

    logMessage(Debug, ": " +
            std::to_string(position.x() * 100) + " " +
            std::to_string(position.y() * 100) + " " +
            std::to_string(position.z() * 100));
    logMessage(Debug, "[Linear Velocity]: " +
            std::to_string(tracker_data[deviceIndex].linear_velocity.x()) + " " +
            std::to_string(tracker_data[deviceIndex].linear_velocity.y()) + " " +
            std::to_string(tracker_data[deviceIndex].linear_velocity.z()));
    logMessage(Debug, "[Angular Velocity]: " +
            std::to_string(tracker_data[deviceIndex].angular_velocity.x()) + " " +
            std::to_string(tracker_data[deviceIndex].angular_velocity.y()) + " " +
            std::to_string(tracker_data[deviceIndex].angular_velocity.z()));
    logMessage(Debug,
               std::string(" Pose Valid: ") +
               (tracker_data[deviceIndex].pose_is_valid ? "True" : "False"));

    // Validate position change for tracker
    if (!validatePositionChange(position, deviceIndex, false)) { // false indicates it's a tracker
        VRUtils::HapticFeedback(pHMD, deviceIndex, VRInputConfig::WARNING_HAPTIC_DURATION);
        // If data is unreasonable, we skip updating prev_position/time for this frame to avoid
        // propagating bad data, and we don't mark it as updated for publishing.
        return;
    }
    // --- NEW: push tracker pose via the normal controller pipe ---
    VRControllerData tracker_as_controller;
    VRUtils::resetJsonData(tracker_as_controller);          // zero everything

    tracker_as_controller.time = Server::getCurrentTimeWithMilliseconds();
    tracker_as_controller.role = VRInputConfig::ROLE_TRACKER0;

    /* pose (reuse helpers already used for controllers) */
    tracker_as_controller.setPosition(position);            // no Y‑offset for tracker
    tracker_as_controller.setQuaternion(quaternion);

    // hand it off exactly like a controller
    {
        std::lock_guard<std::mutex> lk(data_mutex);
        shared_data = tracker_as_controller;
        data_cv.notify_one();
    }

    // Mark this tracker as updated for potential publishing
    tracker_updated[deviceIndex] = true;

    // Note on Tracker Buttons:
    // The Vive Tracker 3.0 has a single button (power button) and pogo pin/USB inputs.
    // These are NOT easily accessible via vr::IVRSystem::GetControllerState.
    // To get input from a generic tracker, you would typically need to use the OpenVR Input system (IVRInput)
    // with action manifests and bindings, which is a more complex setup.
    // For this modification, we are focusing on pose data.
}

// Generalized position validation method for both controllers and trackers
bool ViveInput::validatePositionChange(const Eigen::Vector3d& currentPosition,
                                       uint32_t deviceIndex,
                                       bool isController)
{
    const auto current_time = std::chrono::steady_clock::now();

    // Select the right storage for controllers vs trackers
    Eigen::Vector3d &prev_pos  = isController ?
        prev_position[deviceIndex] : prev_tracker_position[deviceIndex];
    std::chrono::steady_clock::time_point &prev_tp = isController ?
        prev_time[deviceIndex] : prev_tracker_time[deviceIndex];

    // First-sample flags
    bool first_flag;
    if (isController) {
        if (deviceIndex >= VRInputConfig::MAX_CONTROLLERS) {
            logMessage(Error, "Invalid controller role index: " + std::to_string(deviceIndex));
            return false;
        }
        first_flag = first_run[deviceIndex];
    } else {
        if (deviceIndex >= first_tracker_run.size()) {
            logMessage(Error, "Invalid tracker device index: " + std::to_string(deviceIndex));
            return false;
        }
        first_flag = first_tracker_run[deviceIndex];
    }

    const char *device_type_str = isController ? "CONTROLLER" : "TRACKER";

    if (!first_flag) {
        // time delta
        const std::chrono::duration<float> time_diff = current_time - prev_tp;
        const float delta_time = time_diff.count();
        if (delta_time <= 0.f) {
            logMessage(Debug, std::string("[") + device_type_str +
                               " " + std::to_string(deviceIndex) +
                               "] Non-positive dt; skipping.");
            // don't update prev_pos/prev_tp
            return false;
        }

        // distance & velocity
        const Eigen::Vector3d position_change = currentPosition - prev_pos;
        const double delta_distance = position_change.norm();
        const double velocity = delta_distance / delta_time;

        logMessage(Debug, "[" + std::string(device_type_str) + " " + std::to_string(deviceIndex) +
                          "] Velocity: " + std::to_string(velocity) + " units/s");
        logMessage(Debug, "[" + std::string(device_type_str) + " " + std::to_string(deviceIndex) +
                          "] Delta pos: " + std::to_string(delta_distance) + " units");

        // sanity check
        if (!VRTransforms::isPositionChangeReasonable(currentPosition,
                                                      prev_pos,
                                                      VRInputConfig::POSITION_THRESHOLD)) {
            logMessage(Warning,
                       "[" + std::string(device_type_str) + " " + std::to_string(deviceIndex) +
                       "] Unreasonable delta_distance: " + std::to_string(delta_distance) +
                       " units. Skipping this data.");
            return false;
        }
        // OK to publish
        logMessage(Debug, "[" + std::string(device_type_str) + " " + std::to_string(deviceIndex) +
                          "] Will publish this data");
    } else {
        // consume first sample; next time we'll validate
        if (isController) {
            first_run[deviceIndex] = false;
        } else {
            first_tracker_run[deviceIndex] = false;
        }
    }

    // update history (only if we didn't early-return)
    prev_pos = currentPosition;
    prev_tp  = current_time;
    return true;
}

int main(int argc, char **argv) {
    Server::setupSignalHandlers();

    std::mutex data_mutex;
    std::condition_variable data_cv;
    VRControllerData shared_data;
    
    // Configure publishing frequency (default 50Hz, can be adjusted)
    double publish_frequency = VRInputConfig::DEFAULT_PUBLISH_FREQUENCY;
    
    // Check for command line argument to set frequency
    if (argc > 1) {
        try {
            publish_frequency = std::stod(argv[1]);
            if (publish_frequency <= VRInputConfig::MIN_PUBLISH_FREQUENCY || publish_frequency > VRInputConfig::MAX_PUBLISH_FREQUENCY) {
                std::cerr << "Invalid frequency. Using default " << VRInputConfig::DEFAULT_PUBLISH_FREQUENCY << "Hz." << std::endl;
                publish_frequency = VRInputConfig::DEFAULT_PUBLISH_FREQUENCY;
            }
        } catch (const std::exception& e) {
            std::cerr << "Invalid frequency argument. Using default " << VRInputConfig::DEFAULT_PUBLISH_FREQUENCY << "Hz." << std::endl;
            publish_frequency = VRInputConfig::DEFAULT_PUBLISH_FREQUENCY;
        }
    }

    Server server(12345, data_mutex, data_cv, shared_data);
    std::thread serverThread(&Server::start, &server);
    serverThread.detach(); // Detach the server thread

    ViveInput vive_input(data_mutex, data_cv, shared_data, publish_frequency);
    vive_input.runVR();

    return 0;
}
