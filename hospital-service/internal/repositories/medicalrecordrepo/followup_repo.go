package medicalrecordrepo

import (
	"hospital-service/internal/models/medicalrecord"

	"gorm.io/gorm"
)

type FollowUpRepo struct {
	db *gorm.DB
}

func NewFollowUpRepository(db *gorm.DB) *FollowUpRepo {
	return &FollowUpRepo{db: db}
}

func (r *FollowUpRepo) AddFollowUp(fu *medicalrecord.FollowUp) error {
	return r.db.Create(fu).Error
}

func (r *FollowUpRepo) GetByRecordID(recordID string) ([]medicalrecord.FollowUp, error) {
	var res []medicalrecord.FollowUp
	err := r.db.Where("record_id = ?", recordID).Find(&res).Error
	return res, err
}

func (r *FollowUpRepo) DeleteByID(id string) error {
	return r.db.Delete(&medicalrecord.FollowUp{}, "follow_up_id = ?", id).Error
}

func (r *FollowUpRepo) Update(fu *medicalrecord.FollowUp) error {
	return r.db.Save(fu).Error
}
