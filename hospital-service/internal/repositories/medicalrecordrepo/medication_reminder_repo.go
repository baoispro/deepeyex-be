package medicalrecordrepo

import (
	"hospital-service/internal/models/medicalrecord"

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
