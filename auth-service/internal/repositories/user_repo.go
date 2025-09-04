package repositories

import (
	"auth-service/internal/models"

	"gorm.io/gorm"
)

type UserRepo struct{ db *gorm.DB }

func NewUserRepo(db *gorm.DB) *UserRepo { return &UserRepo{db: db} }

func (r *UserRepo) Create(u *models.User) error {
	return r.db.Create(u).Error
}

func (r *UserRepo) FindByUsername(username string) (*models.User, error) {
	var u models.User
	if err := r.db.Where("username = ?", username).First(&u).Error; err != nil {
		return nil, err
	}
	return &u, nil
}

func (r *UserRepo) FindByID(id string) (*models.User, error) {
	var u models.User
	if err := r.db.First(&u, "id = ?", id).Error; err != nil {
		return nil, err
	}
	return &u, nil
}

func (r *UserRepo) FindByFirebaseUID(firebaseUID string) (*models.User, error) {
    var u models.User
    if err := r.db.Where("firebase_uid = ?", firebaseUID).First(&u).Error; err != nil {
        return nil, err
    }
    return &u, nil
}

func (r *UserRepo) FindByEmail(email string) (*models.User, error) {
    var u models.User
    if err := r.db.Where("email = ?", email).First(&u).Error; err != nil {
        return nil, err
    }
    return &u, nil
}