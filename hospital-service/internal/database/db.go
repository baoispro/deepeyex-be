package database

import (
	"hospital-service/internal/config"
	"hospital-service/internal/models/appointment"
	"hospital-service/internal/models/doctor"
	"hospital-service/internal/models/drug"
	"hospital-service/internal/models/hospital"
	"hospital-service/internal/models/order"
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

	// 1. Bảng Hospital (cha) trước
	if err := db.AutoMigrate(&hospital.Hospital{}); err != nil {
		return err
	}

	// 2. Bảng Doctor, phụ thuộc Hospital
	if err := db.AutoMigrate(&doctor.Doctor{}); err != nil {
		return err
	}

	// 3. Bảng Patient và TimeSlot không phụ thuộc bảng khác
	if err := db.AutoMigrate(&patient.Patient{}); err != nil {
		return err
	}
	if err := db.AutoMigrate(&appointment.TimeSlot{}); err != nil {
		return err
	}

	// 4. Bảng Appointment, phụ thuộc Patient và TimeSlot
	if err := db.AutoMigrate(&appointment.Appointment{}); err != nil {
		return err
	}

	// 5. Bảng Drug
	if err := db.AutoMigrate(&drug.Drug{}); err != nil {
		return err
	}

	// 6. Bảng Order và OrderItem
	if err := db.AutoMigrate(&order.Order{}, &order.OrderItem{}); err != nil {
		return err
	}

	return nil
}
