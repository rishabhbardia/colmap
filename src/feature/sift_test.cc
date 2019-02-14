// Copyright (c) 2018, ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

#define TEST_NAME "feature/sift_test"
#include "util/testing.h"


#include "feature/sift.h"
#include "feature/utils.h"
#include "util/math.h"
#include "util/opengl_utils.h"
#include "util/random.h"

using namespace colmap;

void CreateImageWithSquare(const int size, Bitmap* bitmap) {
  bitmap->Allocate(size, size, false);
  bitmap->Fill(BitmapColor<uint8_t>(0, 0, 0));
  for (int r = size / 2 - size / 8; r < size / 2 + size / 8; ++r) {
    for (int c = size / 2 - size / 8; c < size / 2 + size / 8; ++c) {
      bitmap->SetPixel(r, c, BitmapColor<uint8_t>(255));
    }
  }
}

BOOST_AUTO_TEST_CASE(TestExtractSiftFeaturesCPU) {
  Bitmap bitmap;
  CreateImageWithSquare(256, &bitmap);

  FeatureKeypoints keypoints;
  FeatureDescriptors descriptors;
  BOOST_CHECK(ExtractSiftFeaturesCPU(SiftExtractionOptions(), bitmap,
                                     &keypoints, &descriptors));

  BOOST_CHECK_EQUAL(keypoints.size(), 22);
  for (size_t i = 0; i < keypoints.size(); ++i) {
    BOOST_CHECK_GE(keypoints[i].x, 0);
    BOOST_CHECK_GE(keypoints[i].y, 0);
    BOOST_CHECK_LE(keypoints[i].x, bitmap.Width());
    BOOST_CHECK_LE(keypoints[i].y, bitmap.Height());
    BOOST_CHECK_GT(keypoints[i].ComputeScale(), 0);
    BOOST_CHECK_GT(keypoints[i].ComputeOrientation(), -M_PI);
    BOOST_CHECK_LT(keypoints[i].ComputeOrientation(), M_PI);
  }

  BOOST_CHECK_EQUAL(descriptors.rows(), 22);
  for (FeatureDescriptors::Index i = 0; i < descriptors.rows(); ++i) {
    BOOST_CHECK_LT(std::abs(descriptors.row(i).cast<float>().norm() - 512), 1);
  }
}

BOOST_AUTO_TEST_CASE(TestExtractCovariantSiftFeaturesCPU) {
  Bitmap bitmap;
  CreateImageWithSquare(256, &bitmap);

  FeatureKeypoints keypoints;
  FeatureDescriptors descriptors;
  BOOST_CHECK(ExtractCovariantSiftFeaturesCPU(SiftExtractionOptions(), bitmap,
                                              &keypoints, &descriptors));

  BOOST_CHECK_EQUAL(keypoints.size(), 22);
  for (size_t i = 0; i < keypoints.size(); ++i) {
    BOOST_CHECK_GE(keypoints[i].x, 0);
    BOOST_CHECK_GE(keypoints[i].y, 0);
    BOOST_CHECK_LE(keypoints[i].x, bitmap.Width());
    BOOST_CHECK_LE(keypoints[i].y, bitmap.Height());
    BOOST_CHECK_GT(keypoints[i].ComputeScale(), 0);
    BOOST_CHECK_GT(keypoints[i].ComputeOrientation(), -M_PI);
    BOOST_CHECK_LT(keypoints[i].ComputeOrientation(), M_PI);
  }

  BOOST_CHECK_EQUAL(descriptors.rows(), 22);
  for (FeatureDescriptors::Index i = 0; i < descriptors.rows(); ++i) {
    BOOST_CHECK_LT(std::abs(descriptors.row(i).cast<float>().norm() - 512), 1);
  }
}

BOOST_AUTO_TEST_CASE(TestExtractCovariantAffineSiftFeaturesCPU) {
  Bitmap bitmap;
  CreateImageWithSquare(256, &bitmap);

  FeatureKeypoints keypoints;
  FeatureDescriptors descriptors;
  SiftExtractionOptions options;
  options.estimate_affine_shape = true;
  BOOST_CHECK(ExtractCovariantSiftFeaturesCPU(options, bitmap, &keypoints,
                                              &descriptors));

  BOOST_CHECK_EQUAL(keypoints.size(), 10);
  for (size_t i = 0; i < keypoints.size(); ++i) {
    BOOST_CHECK_GE(keypoints[i].x, 0);
    BOOST_CHECK_GE(keypoints[i].y, 0);
    BOOST_CHECK_LE(keypoints[i].x, bitmap.Width());
    BOOST_CHECK_LE(keypoints[i].y, bitmap.Height());
    BOOST_CHECK_GT(keypoints[i].ComputeScale(), 0);
    BOOST_CHECK_GT(keypoints[i].ComputeOrientation(), -M_PI);
    BOOST_CHECK_LT(keypoints[i].ComputeOrientation(), M_PI);
  }

  BOOST_CHECK_EQUAL(descriptors.rows(), 10);
  for (FeatureDescriptors::Index i = 0; i < descriptors.rows(); ++i) {
    BOOST_CHECK_LT(std::abs(descriptors.row(i).cast<float>().norm() - 512), 1);
  }
}

BOOST_AUTO_TEST_CASE(TestExtractCovariantDSPSiftFeaturesCPU) {
  Bitmap bitmap;
  CreateImageWithSquare(256, &bitmap);

  FeatureKeypoints keypoints;
  FeatureDescriptors descriptors;
  SiftExtractionOptions options;
  options.domain_size_pooling = true;
  BOOST_CHECK(ExtractCovariantSiftFeaturesCPU(options, bitmap, &keypoints,
                                              &descriptors));

  BOOST_CHECK_EQUAL(keypoints.size(), 22);
  for (size_t i = 0; i < keypoints.size(); ++i) {
    BOOST_CHECK_GE(keypoints[i].x, 0);
    BOOST_CHECK_GE(keypoints[i].y, 0);
    BOOST_CHECK_LE(keypoints[i].x, bitmap.Width());
    BOOST_CHECK_LE(keypoints[i].y, bitmap.Height());
    BOOST_CHECK_GT(keypoints[i].ComputeScale(), 0);
    BOOST_CHECK_GT(keypoints[i].ComputeOrientation(), -M_PI);
    BOOST_CHECK_LT(keypoints[i].ComputeOrientation(), M_PI);
  }

  BOOST_CHECK_EQUAL(descriptors.rows(), 22);
  for (FeatureDescriptors::Index i = 0; i < descriptors.rows(); ++i) {
    BOOST_CHECK_LT(std::abs(descriptors.row(i).cast<float>().norm() - 512), 1);
  }
}

BOOST_AUTO_TEST_CASE(TestExtractCovariantAffineDSPSiftFeaturesCPU) {
  Bitmap bitmap;
  CreateImageWithSquare(256, &bitmap);

  FeatureKeypoints keypoints;
  FeatureDescriptors descriptors;
  SiftExtractionOptions options;
  options.estimate_affine_shape = true;
  options.domain_size_pooling = true;
  BOOST_CHECK(ExtractCovariantSiftFeaturesCPU(options, bitmap, &keypoints,
                                              &descriptors));

  BOOST_CHECK_EQUAL(keypoints.size(), 10);
  for (size_t i = 0; i < keypoints.size(); ++i) {
    BOOST_CHECK_GE(keypoints[i].x, 0);
    BOOST_CHECK_GE(keypoints[i].y, 0);
    BOOST_CHECK_LE(keypoints[i].x, bitmap.Width());
    BOOST_CHECK_LE(keypoints[i].y, bitmap.Height());
    BOOST_CHECK_GT(keypoints[i].ComputeScale(), 0);
    BOOST_CHECK_GT(keypoints[i].ComputeOrientation(), -M_PI);
    BOOST_CHECK_LT(keypoints[i].ComputeOrientation(), M_PI);
  }

  BOOST_CHECK_EQUAL(descriptors.rows(), 10);
  for (FeatureDescriptors::Index i = 0; i < descriptors.rows(); ++i) {
    BOOST_CHECK_LT(std::abs(descriptors.row(i).cast<float>().norm() - 512), 1);
  }
}


FeatureDescriptors CreateRandomFeatureDescriptors(const size_t num_features) {
  SetPRNGSeed(0);
  Eigen::MatrixXf descriptors(num_features, 128);
  for (size_t i = 0; i < num_features; ++i) {
    for (size_t j = 0; j < 128; ++j) {
      descriptors(i, j) = std::pow(RandomReal(0.0f, 1.0f), 2);
    }
  }
  return FeatureDescriptorsToUnsignedByte(
      L2NormalizeFeatureDescriptors(descriptors));
}

void CheckEqualMatches(const FeatureMatches& matches1,
                       const FeatureMatches& matches2) {
  BOOST_REQUIRE_EQUAL(matches1.size(), matches2.size());
  for (size_t i = 0; i < matches1.size(); ++i) {
    BOOST_CHECK_EQUAL(matches1[i].point2D_idx1, matches2[i].point2D_idx1);
    BOOST_CHECK_EQUAL(matches1[i].point2D_idx2, matches2[i].point2D_idx2);
  }
}



BOOST_AUTO_TEST_CASE(TestMatchSiftFeaturesCPU) {
  const FeatureDescriptors empty_descriptors =
      CreateRandomFeatureDescriptors(0);
  const FeatureDescriptors descriptors1 = CreateRandomFeatureDescriptors(2);
  const FeatureDescriptors descriptors2 = descriptors1.colwise().reverse();

  FeatureMatches matches;

  MatchSiftFeaturesCPU(SiftMatchingOptions(), descriptors1, descriptors2,
                       &matches);
  BOOST_CHECK_EQUAL(matches.size(), 2);
  BOOST_CHECK_EQUAL(matches[0].point2D_idx1, 0);
  BOOST_CHECK_EQUAL(matches[0].point2D_idx2, 1);
  BOOST_CHECK_EQUAL(matches[1].point2D_idx1, 1);
  BOOST_CHECK_EQUAL(matches[1].point2D_idx2, 0);

  MatchSiftFeaturesCPU(SiftMatchingOptions(), empty_descriptors, descriptors2,
                       &matches);
  BOOST_CHECK_EQUAL(matches.size(), 0);
  MatchSiftFeaturesCPU(SiftMatchingOptions(), descriptors1, empty_descriptors,
                       &matches);
  BOOST_CHECK_EQUAL(matches.size(), 0);
  MatchSiftFeaturesCPU(SiftMatchingOptions(), empty_descriptors,
                       empty_descriptors, &matches);
  BOOST_CHECK_EQUAL(matches.size(), 0);
}

BOOST_AUTO_TEST_CASE(TestMatchGuidedSiftFeaturesCPU) {
  FeatureKeypoints empty_keypoints(0);
  FeatureKeypoints keypoints1(2);
  keypoints1[0].x = 1;
  keypoints1[1].x = 2;
  FeatureKeypoints keypoints2(2);
  keypoints2[0].x = 2;
  keypoints2[1].x = 1;
  const FeatureDescriptors empty_descriptors =
      CreateRandomFeatureDescriptors(0);
  const FeatureDescriptors descriptors1 = CreateRandomFeatureDescriptors(2);
  const FeatureDescriptors descriptors2 = descriptors1.colwise().reverse();

  TwoViewGeometry two_view_geometry;
  two_view_geometry.config = TwoViewGeometry::PLANAR_OR_PANORAMIC;
  two_view_geometry.H = Eigen::Matrix3d::Identity();

  MatchGuidedSiftFeaturesCPU(SiftMatchingOptions(), keypoints1, keypoints2,
                             descriptors1, descriptors2, &two_view_geometry);
  BOOST_CHECK_EQUAL(two_view_geometry.inlier_matches.size(), 2);
  BOOST_CHECK_EQUAL(two_view_geometry.inlier_matches[0].point2D_idx1, 0);
  BOOST_CHECK_EQUAL(two_view_geometry.inlier_matches[0].point2D_idx2, 1);
  BOOST_CHECK_EQUAL(two_view_geometry.inlier_matches[1].point2D_idx1, 1);
  BOOST_CHECK_EQUAL(two_view_geometry.inlier_matches[1].point2D_idx2, 0);

  keypoints1[0].x = 100;
  MatchGuidedSiftFeaturesCPU(SiftMatchingOptions(), keypoints1, keypoints2,
                             descriptors1, descriptors2, &two_view_geometry);
  BOOST_CHECK_EQUAL(two_view_geometry.inlier_matches.size(), 1);
  BOOST_CHECK_EQUAL(two_view_geometry.inlier_matches[0].point2D_idx1, 1);
  BOOST_CHECK_EQUAL(two_view_geometry.inlier_matches[0].point2D_idx2, 0);

  MatchGuidedSiftFeaturesCPU(SiftMatchingOptions(), empty_keypoints, keypoints2,
                             empty_descriptors, descriptors2,
                             &two_view_geometry);
  BOOST_CHECK_EQUAL(two_view_geometry.inlier_matches.size(), 0);
  MatchGuidedSiftFeaturesCPU(SiftMatchingOptions(), keypoints1, empty_keypoints,
                             descriptors1, empty_descriptors,
                             &two_view_geometry);
  BOOST_CHECK_EQUAL(two_view_geometry.inlier_matches.size(), 0);
  MatchGuidedSiftFeaturesCPU(SiftMatchingOptions(), empty_keypoints,
                             empty_keypoints, empty_descriptors,
                             empty_descriptors, &two_view_geometry);
  BOOST_CHECK_EQUAL(two_view_geometry.inlier_matches.size(), 0);
}


