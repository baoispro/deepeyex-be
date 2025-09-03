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
func AutoMigrate(db *gorm.DB) error {
	// Thêm tất cả model bạn muốn migrate vào đây
	return db.AutoMigrate(
		&patient.Patient{},
		&doctor.Doctor{},
		&hospital.Hospital{},
	)
}