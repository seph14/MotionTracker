//
//  Transformation.h
//  Motion Tracker
//
//  Created by SEPH LI on 05/03/2025.
//

#ifndef Transformation_h
#define Transformation_h

#include "json.hpp"
#include "CinderOpenCV.h"

struct Transformation {
    cv::Matx33d R;
    cv::Vec3d   t;

    // Construct an identity transformation.
    Transformation() : R(cv::Matx33d::eye()), t(0., 0., 0.) {}

    // Construct from H
    Transformation(const cv::Matx44d &H) : R(H.get_minor<3, 3>(0, 0)), t(H(0, 3), H(1, 3), H(2, 3)) {}

    // Construct from H
    Transformation(const ci::mat4 &H) {
        //mat[col][row]
        R(0, 0) = H[0][0]; R(0, 1) = H[0][1]; R(0, 2) = H[0][2];
        R(1, 0) = H[1][0]; R(1, 1) = H[1][1]; R(1, 2) = H[1][2];
        R(2, 0) = H[2][0]; R(2, 1) = H[2][1]; R(2, 2) = H[2][2];
    
        t(0) = H[3][0]; t(1) = H[3][1]; t(2) = H[3][2];
    }

    // Construct from translation and euler angles
    void from(const ci::vec3& translation, const ci::vec3& euler){
        t            = cv::Vec3d(translation.x, translation.y, translation.z);
        ci::mat3 rot = glm::mat3_cast(glm::quat(glm::radians(euler)));
        R(0, 0) = rot[0][0]; R(0, 1) = rot[0][1]; R(0, 2) = rot[0][2];
        R(1, 0) = rot[1][0]; R(1, 1) = rot[1][1]; R(1, 2) = rot[1][2];
        R(2, 0) = rot[2][0]; R(2, 1) = rot[2][1]; R(2, 2) = rot[2][2];
    }

    // Create homogeneous matrix from this transformation
    cv::Matx44d to_homogeneous() const {
        return cv::Matx44d(
            // row 1
            R(0, 0), R(0, 1), R(0, 2), t(0),
            // row 2
            R(1, 0), R(1, 1), R(1, 2), t(1),
            // row 3
            R(2, 0), R(2, 1), R(2, 2), t(2),
            // row 4
            0, 0, 0, 1);
    }

    nlohmann::json toJson() const {
        using json = nlohmann::json;
        
        json data;
        data["translation"] = { t(0), t(1), t(2) };
        data["rotation"]  = { R(0,0), R(0,1),R(0,2), R(1,0),R(1,1),R(1,2), R(2,0), R(2,1),R(2,2) };
        return data;
    }

    void setFromJson(const nlohmann::json& data) {
        auto translation = data["translation"];
        t(0) = (double)translation[0] / 1000.f;
        t(1) = (double)translation[1] / 1000.f;
        t(2) = (double)translation[2] / 1000.f; //mm -> m

        auto rotation = data["rotation"];
        R(0, 0) = rotation[0];
        R(0, 1) = rotation[1];
        R(0, 2) = rotation[2];
        R(1, 0) = rotation[3];
        R(1, 1) = rotation[4];
        R(1, 2) = rotation[5];
        R(2, 0) = rotation[6];
        R(2, 1) = rotation[7];
        R(2, 2) = rotation[8];
    }
    
    ci::mat4 rotation() const {
        return ci::mat4(
            // row 1
            R(0, 0), R(0, 1), R(0, 2), 0,
            // row 2
            R(1, 0), R(1, 1), R(1, 2), 0,
            // row 3
            R(2, 0), R(2, 1), R(2, 2), 0,
            // row 4
            0,       0,       0,       1);
    }
    
    // Create homogeneous matrix from this transformation
    ci::mat4 to_model(const float scale, const bool flip = false) const {
        glm::mat4 rot = rotation();
        float dir = flip ? 1.f : -1.f;
        return glm::translate( dir * scale * ci::vec3(t(0), t(1), t(2)) / 2500.f ) * glm::inverse(rot);
    }

    ci::mat4 to_touch() {
        glm::mat4 rot = rotation();
        return glm::translate(ci::vec3(t(0), t(1), t(2))) * rot;
    }
    
    // Construct a transformation equivalent to this transformation followed by the second transformation
    Transformation compose_with(const Transformation &second_transformation) const {
        // get this transform
        cv::Matx44d H_1 = to_homogeneous();
        // get the transform to be composed with this one
        cv::Matx44d H_2 = second_transformation.to_homogeneous();
        // get the combined transform
        cv::Matx44d H_3 = H_1 * H_2;
        return Transformation(H_3);
    }
};

#endif /* Transformation_h */
