package medicalrecordrepo

import (
	"hospital-service/internal/models/medicalrecord"
	"time"

	"github.com/google/uuid"
	"gorm.io/gorm"
)

type MedicalRecordRepo struct {
	db *gorm.DB
}

func NewMedicalRecordRepository(db *gorm.DB) *MedicalRecordRepo {
	return &MedicalRecordRepo{db: db}
}

// InitRecordAndDiagnosis: tạo MedicalRecord và AIDiagnosis trong cùng 1 transaction
func (r *MedicalRecordRepo) InitRecord(patientID, appointmentID, doctorID string) (*medicalrecord.MedicalRecord, error) {
	var record *medicalrecord.MedicalRecord

	err := r.db.Transaction(func(tx *gorm.DB) error {
		// Step 1: Tạo MedicalRecord mới
		record = &medicalrecord.MedicalRecord{
			RecordID:      uuid.New().String(),
			PatientID:     patientID,
			AppointmentID: appointmentID,
			DoctorID:      doctorID,
			CreatedAt:     time.Now(),
			UpdatedAt:     time.Now(),
		}

		if err := tx.Create(record).Error; err != nil {
			return err
		}

		return nil
	})

	if err != nil {
		return nil, err
	}

	return record, nil
}

// ---------------- Create record ----------------
func (r *MedicalRecordRepo) Create(record *medicalrecord.MedicalRecord) error {
	return r.db.Create(record).Error
}

// ---------------- Get record by ID ----------------
func (r *MedicalRecordRepo) GetByID(id string) (*medicalrecord.MedicalRecord, error) {
	var record medicalrecord.MedicalRecord
	err := r.db.Preload("Attachments").
		Preload("Prescriptions").
		Preload("AI_Diagnoses").
		First(&record, "record_id = ?", id).Error
	return &record, err
}

// ---------------- List all records ----------------
func (r *MedicalRecordRepo) List() ([]*medicalrecord.MedicalRecord, error) {
	var records []*medicalrecord.MedicalRecord
	err := r.db.
		Preload("Attachments").
		Find(&records).Error
	return records, err
}

// ---------------- Update record ----------------
func (r *MedicalRecordRepo) Update(record *medicalrecord.MedicalRecord) error {
	return r.db.Save(record).Error
}

// ---------------- Delete record ----------------
func (r *MedicalRecordRepo) Delete(id string) error {
	return r.db.Delete(&medicalrecord.MedicalRecord{}, "record_id = ?", id).Error
}

// ---------------- Get record by AppointmentID ----------------
func (r *MedicalRecordRepo) GetByAppointmentID(appointmentID string) (*medicalrecord.MedicalRecord, error) {
	var record medicalrecord.MedicalRecord
	err := r.db.
		Preload("Attachments").
		Preload("Prescriptions").
		Preload("AI_Diagnoses").
		First(&record, "appointment_id = ?", appointmentID).Error
	if err != nil {
		return nil, err
	}
	return &record, nil
}

// ---------------- Get all records by PatientID ----------------
func (r *MedicalRecordRepo) GetByPatientID(patientID string) ([]*medicalrecord.MedicalRecord, error) {
	var records []*medicalrecord.MedicalRecord
	
	// Đơn giản hóa - chỉ lấy records theo patient_id, sắp xếp theo ngày tạo giảm dần
	err := r.db.
		Preload("Attachments").
		Preload("Prescriptions.Items").  // Preload cả Items của Prescriptions
		Preload("AI_Diagnoses").
		Where("patient_id = ?", patientID).
		Order("created_at DESC").  // Sắp xếp theo ngày tạo giảm dần (mới nhất trước)
		Find(&records).Error

	if err != nil {
		return nil, err
	}
	
	return records, nil
}
