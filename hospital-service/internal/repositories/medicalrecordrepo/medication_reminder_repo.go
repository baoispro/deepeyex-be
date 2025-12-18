package medicalrecordrepo

import (
	"hospital-service/internal/models/medicalrecord"
	"time"

	"gorm.io/gorm"
)

type MedicationReminderRepository struct {
	db *gorm.DB
}

func NewMedicationReminderRepository(db *gorm.DB) *MedicationReminderRepository {
	return &MedicationReminderRepository{db: db}
}

func (r *MedicationReminderRepository) Create(reminder *medicalrecord.MedicationReminder) error {
	return r.db.Create(reminder).Error
}

func (r *MedicationReminderRepository) FindByItemID(itemID string) ([]medicalrecord.MedicationReminder, error) {
	var reminders []medicalrecord.MedicationReminder
	err := r.db.Where("prescription_item_id = ?", itemID).Find(&reminders).Error
	return reminders, err
}

func (r *MedicationReminderRepository) UpdateStatus(id string, status string) error {
	return r.db.Model(&medicalrecord.MedicationReminder{}).
		Where("id = ?", id).
		Update("status", status).Error
}

func (r *MedicationReminderRepository) DeleteByItemID(itemID string) error {
	return r.db.Where("prescription_item_id = ?", itemID).Delete(&medicalrecord.MedicationReminder{}).Error
}

func (r *MedicationReminderRepository) FindByPatientID(patientID string) ([]medicalrecord.MedicationReminder, error) {
	var reminders []medicalrecord.MedicationReminder
	err := r.db.
		Table("medication_reminders").
		Joins("INNER JOIN prescription_items ON medication_reminders.prescription_item_id = prescription_items.item_id").
		Joins("INNER JOIN prescriptions ON prescription_items.prescription_id = prescriptions.prescription_id").
		Where("prescriptions.patient_id = ?", patientID).
		Find(&reminders).Error
	return reminders, err
}

func (r *MedicationReminderRepository) FindByPatientIDWithItem(patientID string) ([]medicalrecord.MedicationReminderWithItem, error) {
	var results []medicalrecord.MedicationReminderWithItem
	
	// Lấy ngày hôm nay (chỉ lấy phần date, bỏ qua time)
	today := time.Now()
	startOfDay := time.Date(today.Year(), today.Month(), today.Day(), 0, 0, 0, 0, today.Location())
	endOfDay := startOfDay.Add(24 * time.Hour)
	
	err := r.db.
		Table("medication_reminders AS mr").
		Select(`
			mr.id,
			mr.prescription_item_id,
			mr.reminder_time,
			mr.status,
			mr.created_at,
			pi.drug_name,
			pi.dosage,
			pi.frequency,
			pi.notes
		`).
		Joins("INNER JOIN prescription_items AS pi ON mr.prescription_item_id = pi.item_id").
		Joins("INNER JOIN prescriptions AS p ON pi.prescription_id = p.prescription_id").
		Where("p.patient_id = ?", patientID).
		Where("mr.reminder_time >= ? AND mr.reminder_time < ?", startOfDay, endOfDay).
		Order("mr.reminder_time ASC").
		Scan(&results).Error
	
	return results, err
}
