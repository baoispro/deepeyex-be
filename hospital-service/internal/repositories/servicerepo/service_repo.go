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

// ============== SERVICE CRUD ==============

// CreateService - Tạo service mới
func (r *ServiceRepo) CreateService(s *service.Service) error {
	return r.db.Create(s).Error
}

// GetServiceByID - Lấy service theo ID
func (r *ServiceRepo) GetServiceByID(serviceID string) (*service.Service, error) {
	var s service.Service
	err := r.db.Where("service_id = ?", serviceID).First(&s).Error
	if err != nil {
		return nil, err
	}
	return &s, nil
}

// ListAllServices - Lấy tất cả services
func (r *ServiceRepo) ListAllServices() ([]service.Service, error) {
	var services []service.Service
	err := r.db.Find(&services).Error
	return services, err
}

// UpdateService - Cập nhật service
func (r *ServiceRepo) UpdateService(s *service.Service) error {
	return r.db.Save(s).Error
}

// DeleteService - Xóa service
func (r *ServiceRepo) DeleteService(serviceID string) error {
	return r.db.Delete(&service.Service{}, "service_id = ?", serviceID).Error
}

// ============== DOCTOR-SERVICE RELATIONSHIP ==============

// GetServicesByDoctorID - Lấy tất cả services theo doctor_id
func (r *ServiceRepo) GetServicesByDoctorID(doctorID string) ([]service.Service, error) {
	var services []service.Service

	err := r.db.Table("services").
		Select("services.service_id, services.name, services.duration, services.price, services.created_at, services.updated_at").
		Joins("INNER JOIN doctor_services ds ON ds.service_id = services.service_id").
		Where("ds.doctor_id = ?", doctorID).
		Find(&services).Error

	if err != nil {
		return nil, err
	}

	return services, nil
}

// AssignServiceToDoctor - Gán service cho bác sĩ
func (r *ServiceRepo) AssignServiceToDoctor(doctorID, serviceID string) error {
	doctorService := &service.DoctorService{
		DoctorID:  doctorID,
		ServiceID: serviceID,
	}
	return r.db.Create(doctorService).Error
}

// RemoveServiceFromDoctor - Xóa service khỏi bác sĩ
func (r *ServiceRepo) RemoveServiceFromDoctor(doctorID, serviceID string) error {
	return r.db.Where("doctor_id = ? AND service_id = ?", doctorID, serviceID).
		Delete(&service.DoctorService{}).Error
}

// CheckServiceAssigned - Kiểm tra xem service đã được gán cho bác sĩ chưa
func (r *ServiceRepo) CheckServiceAssigned(doctorID, serviceID string) (bool, error) {
	var count int64
	err := r.db.Model(&service.DoctorService{}).
		Where("doctor_id = ? AND service_id = ?", doctorID, serviceID).
		Count(&count).Error
	return count > 0, err
}