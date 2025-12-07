package services

import (
	"auth-service/internal/enums"
	"auth-service/internal/models"
	"auth-service/internal/repositories"
	"errors"
	"time"

	"github.com/google/uuid"
	"golang.org/x/crypto/bcrypt"
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
	hashedPwd, _ := bcrypt.GenerateFromPassword([]byte(password), bcrypt.DefaultCost)
	u := &models.User{
		ID:          uuid.NewString(),
		Username:    username,
		Email:       email,
		Password:    string(hashedPwd),
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
	if password, ok := updates["password"].(string); ok {
		hashedPwd, _ := bcrypt.GenerateFromPassword([]byte(password), bcrypt.DefaultCost)
		u.Password = string(hashedPwd)
	}
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

// List users with optional filters
func (s *UserService) ListUsers(name, role string) ([]models.User, error) {
	return s.userRepo.FindWithFilters(name, role)
}

// Update password by email
func (s *UserService) UpdatePassword(email, newPassword string) error {
	u, err := s.userRepo.FindByEmail(email)
	if err != nil {
		return err
	}

	if len([]rune(newPassword)) < 8 {
		return errors.New("password must be at least 8 characters long")
	}

	if !isValidPassword(newPassword) {
		return errors.New("password must contain at least 1 uppercase, 1 lowercase, 1 digit, and 1 special character")
	}

	// hash password trước khi lưu
	hashedPwd, _ := bcrypt.GenerateFromPassword([]byte(newPassword), bcrypt.DefaultCost)
	u.Password = string(hashedPwd)
	u.UpdatedAt = time.Now()

	return s.userRepo.Update(u)
}
