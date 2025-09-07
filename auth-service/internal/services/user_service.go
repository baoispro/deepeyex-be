package services

import (
	"auth-service/internal/enums"
	"auth-service/internal/models"
	"auth-service/internal/repositories"
	"errors"
	"time"

	"github.com/google/uuid"
)

type UserService struct {
	userRepo *repositories.UserRepo
}

func NewUserService(ur *repositories.UserRepo) *UserService {
	return &UserService{userRepo: ur}
}

// Create user
func (s *UserService) CreateUser(username, email, password, firebaseUID, role string) (*models.User, error) {
	if username == "" || email == "" {
		return nil, errors.New("username and email are required")
	}
	u := &models.User{
		ID:          uuid.NewString(),
		Username:    username,
		Email:       email,
		Password:    password,
		FirebaseUID: firebaseUID,
		Role:        enums.Role(role),
		CreatedAt:   time.Now(),
		UpdatedAt:   time.Now(),
	}
	if err := s.userRepo.Create(u); err != nil {
		return nil, err
	}
	return u, nil
}

// Get user by ID
func (s *UserService) GetUser(id string) (*models.User, error) {
	return s.userRepo.FindByID(id)
}

// Update user
func (s *UserService) UpdateUser(id string, updates map[string]interface{}) (*models.User, error) {
	u, err := s.userRepo.FindByID(id)
	if err != nil {
		return nil, err
	}
	// cập nhật các trường
	if name, ok := updates["username"].(string); ok {
		u.Username = name
	}
	if email, ok := updates["email"].(string); ok {
		u.Email = email
	}
	if role, ok := updates["role"].(string); ok {
		u.Role = enums.Role(role)
	}
	u.UpdatedAt = time.Now()
	if err := s.userRepo.Update(u); err != nil {
		return nil, err
	}
	return u, nil
}

// Delete user
func (s *UserService) DeleteUser(id string) error {
	return s.userRepo.Delete(id)
}

// List users
func (s *UserService) ListUsers() ([]models.User, error) {
	return s.userRepo.FindAll()
}
