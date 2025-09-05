package database

import (
	"hospital-service/internal/config"
	"hospital-service/internal/models/doctor"
	"hospital-service/internal/models/hospital"
	"hospital-service/internal/models/patient"
	"log"

	"gorm.io/driver/postgres"
	"gorm.io/gorm"
)

func Connect(cfg config.Config) *gorm.DB {
	db, err := gorm.Open(postgres.Open(cfg.DBUrl), &gorm.Config{})
	if err != nil {
		log.Fatal("failed to connect db: ", err)
	}
	return db
}

// AutoMigrate tự động tạo hoặc cập nhật bảng trong database
// AutoMigrate tự động tạo hoặc cập nhật bảng trong database
func AutoMigrate(db *gorm.DB) error {
    // 1. Migrate bảng 'cha' (parent table) trước: Hospital
    err := db.AutoMigrate(&hospital.Hospital{})
    if err != nil {
        return err
    }

    // 2. Migrate bảng 'con' (child table) sau: Doctor phụ thuộc vào Hospital
    err = db.AutoMigrate(&doctor.Doctor{})
    if err != nil {
        return err
    }

    // 3. Migrate các bảng không có mối quan hệ phụ thuộc lẫn nhau
    err = db.AutoMigrate(&patient.Patient{})
    if err != nil {
        return err
    }

    return nil
}