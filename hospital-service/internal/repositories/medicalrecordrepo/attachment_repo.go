package medicalrecordrepo

import (
	"hospital-service/internal/models/medicalrecord"

	"gorm.io/gorm"
)


type AttachmentRepo struct {
	db *gorm.DB
}

func NewAttachmentRepository(db *gorm.DB) *AttachmentRepo {
	return &AttachmentRepo{db: db}
}

func (r *AttachmentRepo) AddAttachment(att *medicalrecord.Attachment) error {
	return r.db.Create(att).Error
}

func (r *AttachmentRepo) GetByRecordID(recordID string) ([]medicalrecord.Attachment, error) {
	var res []medicalrecord.Attachment
	err := r.db.Where("record_id = ?", recordID).Find(&res).Error
	return res, err
}

func (r *AttachmentRepo) DeleteByID(id string) error {
	return r.db.Delete(&medicalrecord.Attachment{}, "id = ?", id).Error
}


