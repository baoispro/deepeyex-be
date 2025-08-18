package services

import (
	"user-service/internal/models"
	"user-service/internal/repositories"

	"golang.org/x/crypto/bcrypt"
)

func RegisterUser(email, password string) error {
	hashed, _ := bcrypt.GenerateFromPassword([]byte(password), bcrypt.DefaultCost)
	user := models.User{Email: email, Password: string(hashed)}
	return repositories.CreateUser(&user)
}

func LoginUser(email, password string) bool {
	user, err := repositories.GetUserByEmail(email)
	if err != nil {
		return false
	}
	return bcrypt.CompareHashAndPassword([]byte(user.Password), []byte(password)) == nil
}
