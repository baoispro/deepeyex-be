package database

import (
	"hospital-service/internal/config"
	"hospital-service/internal/models/appointment"
	"hospital-service/internal/models/doctor"
	"hospital-service/internal/models/drug"
	"hospital-service/internal/models/hospital"
	"hospital-service/internal/models/medicalrecord"
	"hospital-service/internal/models/order"
	"hospital-service/internal/models/patient"
	"hospital-service/internal/models/service"
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
	// Enum AppointmentStatus
	if err := db.Exec(`DO $$
    BEGIN
        IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'appointment_status') THEN
            CREATE TYPE appointment_status AS ENUM ('PENDING','CONFIRMED','COMPLETED','CANCELED');
        END IF;
    END$$;`).Error; err != nil {
		return err
	}

	// Enum OrderStatus
	if err := db.Exec(`DO $$
    BEGIN
        IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'order_status') THEN
            CREATE TYPE order_status AS ENUM ('PENDING', 'PAID', 'CANCELED', 'DELIVERED');
        END IF;
    END$$;`).Error; err != nil {
		return err
	}

	if err := db.AutoMigrate(&hospital.Hospital{}); err != nil {
		return err
	}

	if err := db.AutoMigrate(&doctor.Doctor{}); err != nil {
		return err
	}

	if err := db.AutoMigrate(&patient.Patient{}); err != nil {
		return err
	}

	// tạo bảng Appointment trước
	if err := db.AutoMigrate(&appointment.Appointment{}); err != nil {
		return err
	}

	// tạo bảng TimeSlot sau, đảm bảo Appointment tồn tại
	if err := db.AutoMigrate(&appointment.TimeSlot{}); err != nil {
		return err
	}

	if err := db.AutoMigrate(&drug.Drug{}); err != nil {
		return err
	}

	if err := db.AutoMigrate(
		&service.Service{},
	); err != nil {
		return err
	}

	if err := db.AutoMigrate(&order.Order{}, &order.OrderItem{}); err != nil {
		return err
	}

	if err := db.AutoMigrate(
		&medicalrecord.MedicalRecord{},
		&medicalrecord.AIDiagnosis{},
		&medicalrecord.AIRecommendedPlan{},
	); err != nil {
		return err
	}

	if err := db.AutoMigrate(
		&medicalrecord.Prescription{},
		&medicalrecord.PrescriptionItem{},
	); err != nil {
		return err
	}

	// --- 9. Attachments ---
	if err := db.AutoMigrate(&medicalrecord.Attachment{}); err != nil {
		return err
	}

	// --- 10. FollowUps ---
	if err := db.AutoMigrate(&medicalrecord.FollowUp{}); err != nil {
		return err
	}

	if err := db.AutoMigrate(
		&service.DoctorService{},
	); err != nil {
		return err
	}

	return nil
}
