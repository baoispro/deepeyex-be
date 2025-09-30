package servicerepo

import (
	"hospital-service/internal/models/service"

	"gorm.io/gorm"
)

type ServiceRepo struct {
	db *gorm.DB
}

func NewServiceRepo(db *gorm.DB) *ServiceRepo {
	return &ServiceRepo{db: db}
}

// GetServicesByDoctorID - lấy tất cả service theo doctor_id

// GetServicesByDoctorID - Lấy tất cả services theo doctor_id
func (r *ServiceRepo) GetServicesByDoctorID(doctorID string) ([]service.Service, error) {
    var services []service.Service

    err := r.db.Table("services").
        Select("services.service_id, services.name, services.duration, services.price").
        Joins("INNER JOIN doctor_services ds ON ds.service_id = services.service_id").
        Where("ds.doctor_id = ?", doctorID).
        Find(&services).Error

    if err != nil {
        return nil, err
    }

    return services, nil
}