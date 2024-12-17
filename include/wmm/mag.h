#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "wmm/GeomagnetismLibrary.hpp"

bool calc_local_mag_field(const Eigen::Vector3d &gps, Eigen::Vector3d& mag_ENU, const std::string& wmm_cof_path) {
    std::shared_ptr<WMMProcess> mag_proc(new WMMProcess());

    MAGtype_MagneticModel *MagneticModel;
    MAGtype_Geoid Geoid;
    MAGtype_Ellipsoid Ellip;
    MAGtype_CoordGeodetic CoordGeodetic;
    MAGtype_Date UserDate;
    MAGtype_GeoMagneticElements GeoMagneticElements;
    MAGtype_CoordSpherical CoordSpherical;

    mag_proc->MAG_SetDefaults(&Ellip, &Geoid);

    mag_proc->MAG_robustReadMagModels(const_cast<char *>(wmm_cof_path.c_str()), (MAGtype_MagneticModel * (*)[]) & MagneticModel, 1);
    if (MagneticModel == NULL) {
        std::cerr << "Error loading WMM model." << std::endl;
        return false;
    }

    time_t rawtime = time(NULL);
    struct tm *timeinfo = localtime(&rawtime);
    UserDate.Year = timeinfo->tm_year + 1900;
    UserDate.Month = timeinfo->tm_mon + 1;
    UserDate.Day = timeinfo->tm_mday;
    UserDate.DecimalYear =
    UserDate.Year + (UserDate.Month - 1) / 12.0 + (UserDate.Day - 1) / 365.25;

    CoordGeodetic.phi = gps[0];                   // latitude
    CoordGeodetic.lambda = gps[1];                // longitude
    CoordGeodetic.HeightAboveEllipsoid = gps[2];  // altitude

    mag_proc->MAG_GeodeticToSpherical(Ellip, CoordGeodetic, &CoordSpherical);
    mag_proc->MAG_Geomag(Ellip, CoordSpherical, CoordGeodetic, MagneticModel, &GeoMagneticElements);

    mag_ENU[0] =  GeoMagneticElements.Y* 1e-3;
    mag_ENU[1] =  GeoMagneticElements.X* 1e-3;
    mag_ENU[2] = -GeoMagneticElements.Z* 1e-3;

    mag_proc->MAG_FreeMagneticModelMemory(MagneticModel);
    return true;
}

// double Decl; /* 1. Angle between the magnetic field vector and true north, positive east*/
// double Incl; /*2. Angle between the magnetic field vector and the horizontal plane, positive down*/
// double F; /*3. Magnetic Field Strength*/
// double H; /*4. Horizontal Magnetic Field Strength*/
// double X; /*5. Northern component of the magnetic field vector*/
// double Y; /*6. Eastern component of the magnetic field vector*/
// double Z; /*7. Downward component of the magnetic field vector*/