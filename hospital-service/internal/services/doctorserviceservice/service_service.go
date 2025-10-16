package doctorserviceservice

import (
	"errors"
	"hospital-service/internal/models/service"
	"hospital-service/internal/repositories/servicerepo"
	"time"

	"github.com/google/uuid"
)

type ServiceService struct {
	repo *servicerepo.ServiceRepo
}

func NewServiceService(repo *servicerepo.ServiceRepo) *ServiceService {
	return &ServiceService{repo: repo}
}

// ============== SERVICE CRUD ==============

// CreateService - Tạo service mới
func (s *ServiceService) CreateService(name string, duration int, price float64) (*service.Service, error) {
	if name == "" || duration <= 0 || price < 0 {
		return nil, errors.New("invalid service data")
	}

	newService := &service.Service{
		ServiceID: uuid.New().String(),
		Name:      name,
		Duration:  duration,
		Price:     price,
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
	}

	err := s.repo.CreateService(newService)
	if err != nil {
		return nil, err
	}

	return newService, nil
}

// GetServiceByID - Lấy service theo ID
func (s *ServiceService) GetServiceByID(serviceID string) (*service.Service, error) {
	return s.repo.GetServiceByID(serviceID)
}

// ListAllServices - Lấy tất cả services
func (s *ServiceService) ListAllServices(name, duration string) ([]service.Service, error) {
	return s.repo.FindWithFilters(name, duration)
}

// UpdateService - Cập nhật service
func (s *ServiceService) UpdateService(serviceID, name string, duration int, price float64) (*service.Service, error) {
	// Get existing service
	existingService, err := s.repo.GetServiceByID(serviceID)
	if err != nil {
		return nil, err
	}

	// Update fields
	if name != "" {
		existingService.Name = name
	}
	if duration > 0 {
		existingService.Duration = duration
	}
	if price >= 0 {
		existingService.Price = price
	}
	existingService.UpdatedAt = time.Now()

	err = s.repo.UpdateService(existingService)
	if err != nil {
		return nil, err
	}

	return existingService, nil
}

// DeleteService - Xóa service
func (s *ServiceService) DeleteService(serviceID string) error {
	return s.repo.DeleteService(serviceID)
}

// ============== DOCTOR-SERVICE RELATIONSHIP ==============

// GetServicesByDoctorID - Lấy tất cả service theo doctor_id
func (s *ServiceService) GetServicesByDoctorID(doctorID string) ([]service.Service, error) {
	return s.repo.GetServicesByDoctorID(doctorID)
}

// AssignServiceToDoctor - Gán service cho bác sĩ
func (s *ServiceService) AssignServiceToDoctor(doctorID, serviceID string) error {
	// Check if already assigned
	isAssigned, err := s.repo.CheckServiceAssigned(doctorID, serviceID)
	if err != nil {
		return err
	}
	if isAssigned {
		return errors.New("service already assigned to this doctor")
	}

	// Assign service
	return s.repo.AssignServiceToDoctor(doctorID, serviceID)
}

// RemoveServiceFromDoctor - Xóa service khỏi bác sĩ
func (s *ServiceService) RemoveServiceFromDoctor(doctorID, serviceID string) error {
	return s.repo.RemoveServiceFromDoctor(doctorID, serviceID)
}
