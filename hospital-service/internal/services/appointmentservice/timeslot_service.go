package appointmentservice

import (
	"errors"
	"fmt"
	"hospital-service/internal/models/appointment"
	"hospital-service/internal/models/doctor"
	"hospital-service/internal/repositories/appointmentrepo"
	"hospital-service/internal/repositories/doctorrepo"
	"log"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/xuri/excelize/v2"
)

type Doctor = doctor.Doctor

type TimeSlotService struct {
	timeSlotRepo *appointmentrepo.TimeSlotRepo
	doctorRepo   *doctorrepo.DoctorRepo
}

func NewTimeSlotService(timeSlotRepo *appointmentrepo.TimeSlotRepo, doctorRepo *doctorrepo.DoctorRepo) *TimeSlotService {
	return &TimeSlotService{
		timeSlotRepo: timeSlotRepo,
		doctorRepo:   doctorRepo,
	}
}

type TimeSlotInput struct {
	StartTime time.Time
	EndTime   time.Time
	Capacity  int
}

type CreateMultiShiftSlotsRequest struct {
    DoctorID string            `json:"doctor_id" binding:"required"`
    Shifts   []ShiftSelection  `json:"shifts" binding:"required"`
}

type ShiftSelection struct {
    Date  string   `json:"date" binding:"required"`  // YYYY-MM-DD
    Slots []string `json:"slots" binding:"required"` // ["morning","afternoon","evening"]
}


//
// ---------------- CRUD ----------------
//

// Create tạo 1 time slot thủ công
func (s *TimeSlotService) Create(doctorID string, startTime, endTime time.Time, capacity int) (*appointment.TimeSlot, error) {
	if doctorID == "" {
		return nil, errors.New("doctor ID is required")
	}
	if capacity <= 0 {
		return nil, errors.New("capacity must be greater than 0")
	}
	if !startTime.Before(endTime) {
		return nil, errors.New("startTime must be before endTime")
	}

	// Kiểm tra trùng
	count, err := s.timeSlotRepo.CountOverlapping(doctorID, startTime, endTime)
	if err != nil {
		return nil, fmt.Errorf("failed to check overlapping slots: %v", err)
	}
	if count > 0 {
		return nil, errors.New("doctor already has a timeslot in this time range")
	}

	slot := &appointment.TimeSlot{
		SlotID:    generateSlotID(),
		DoctorID:  doctorID,
		StartTime: startTime,
		EndTime:   endTime,
		Capacity:  capacity,
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
	}

	if err := s.timeSlotRepo.Create(slot); err != nil {
		return nil, err
	}
	return slot, nil
}

func (s *TimeSlotService) GetByID(id string) (*appointment.TimeSlot, error) {
	return s.timeSlotRepo.FindByID(id)
}

func (s *TimeSlotService) GetByDoctorID(doctorID string) ([]appointment.TimeSlot, error) {
	return s.timeSlotRepo.FindByDoctorID(doctorID)
}

func (s *TimeSlotService) Update(slotID string, startTime, endTime *time.Time, capacity *int) (*appointment.TimeSlot, error) {
	slot, err := s.timeSlotRepo.FindByID(slotID)
	if err != nil {
		return nil, fmt.Errorf("timeslot not found: %v", err)
	}

	if startTime != nil {
		slot.StartTime = *startTime
	}
	if endTime != nil {
		slot.EndTime = *endTime
	}
	if capacity != nil {
		if *capacity <= 0 {
			return nil, errors.New("capacity must be greater than 0")
		}
		slot.Capacity = *capacity
	}

	if !slot.StartTime.Before(slot.EndTime) {
		return nil, errors.New("startTime must be before endTime")
	}

	count, err := s.timeSlotRepo.CountOverlapping(slot.DoctorID, slot.StartTime, slot.EndTime)
	if err != nil {
		return nil, fmt.Errorf("failed to check overlapping slots: %v", err)
	}
	if count > 0 {
		return nil, errors.New("doctor already has a timeslot in this time range")
	}

	slot.UpdatedAt = time.Now()
	if err := s.timeSlotRepo.Update(slot); err != nil {
		return nil, err
	}

	return slot, nil
}

func (s *TimeSlotService) Delete(id string) error {
	return s.timeSlotRepo.Delete(id)
}

func (s *TimeSlotService) ListAll() ([]appointment.TimeSlot, error) {
	return s.timeSlotRepo.ListAll()
}

//
// ---------------- Generate ----------------
//

// GenerateTimeSlotsForWeek tạo time slots cho tất cả bác sĩ trong tuần tới (thứ 2 đến thứ 6)
func (s *TimeSlotService) GenerateTimeSlotsForWeek() error {
	doctors, err := s.doctorRepo.List()
	if err != nil {
		return err
	}

	nextWeekStart := s.getNextWeekStart(time.Now())

	// Chỉ tạo time slots cho thứ 2 đến thứ 6 (5 ngày làm việc)
	for i := 0; i < 5; i++ {
		date := nextWeekStart.AddDate(0, 0, i)
		if err := s.generateTimeSlotsForDate(doctors, date); err != nil {
			return err
		}
	}
	log.Printf("✅ Generated time slots for all doctors for week starting %v (Monday to Friday)", nextWeekStart)
	return nil
}

// GenerateTimeSlotsForDoctor tạo slot tuần cho 1 bác sĩ (thứ 2 đến thứ 6)
func (s *TimeSlotService) GenerateTimeSlotsForDoctor(doctorID string) error {
	doctor, err := s.doctorRepo.FindByID(doctorID)
	if err != nil {
		return err
	}

	nextWeekStart := s.getNextWeekStart(time.Now())

	// Chỉ tạo time slots cho thứ 2 đến thứ 6 (5 ngày làm việc)
	for i := 0; i < 5; i++ {
		date := nextWeekStart.AddDate(0, 0, i)
		if err := s.generateTimeSlotsForDate([]Doctor{*doctor}, date); err != nil {
			return err
		}
	}
	return nil
}

//
// ---------------- Helper ----------------
//

func (s *TimeSlotService) generateTimeSlotsForDate(doctors []Doctor, date time.Time) error {
	for _, doctor := range doctors {
		startOfDay := time.Date(date.Year(), date.Month(), date.Day(), 0, 0, 0, 0, date.Location())
		endOfDay := startOfDay.Add(24 * time.Hour)

		// Xóa slot cũ
		if err := s.timeSlotRepo.DeleteByDoctorIDAndDateRange(doctor.DoctorID, startOfDay, endOfDay); err != nil {
			continue
		}

		// Tạo slot mới
		timeSlots := s.createTimeSlotsForDoctor(doctor.DoctorID, date)
		if len(timeSlots) > 0 {
			if err := s.timeSlotRepo.CreateBatch(timeSlots); err != nil {
				continue
			}
		}
	}
	return nil
}

func (s *TimeSlotService) createTimeSlotsForDoctor(doctorID string, date time.Time) []appointment.TimeSlot {
	var timeSlots []appointment.TimeSlot

	shifts := []struct {
		startHour, startMinute int
		endHour, endMinute     int
	}{
		{8, 30, 9, 0},
		{9, 0, 10, 0},
		{10, 0, 11, 0},
		{11, 0, 12, 0},
		{13, 30, 14, 0},
		{14, 0, 15, 0},
		{15, 0, 16, 0},
		{16, 0, 17, 0},
		{18, 0, 19, 0},
		{19, 0, 20, 0},
		{20, 0, 21, 0},
	}

	for _, shift := range shifts {
		startTime := time.Date(date.Year(), date.Month(), date.Day(), shift.startHour, shift.startMinute, 0, 0, date.Location())
		endTime := time.Date(date.Year(), date.Month(), date.Day(), shift.endHour, shift.endMinute, 0, 0, date.Location())

		timeSlot := appointment.TimeSlot{
			SlotID:    generateSlotID(),
			DoctorID:  doctorID,
			StartTime: startTime,
			EndTime:   endTime,
			Capacity:  1,
			CreatedAt: time.Now(),
			UpdatedAt: time.Now(),
		}
		timeSlots = append(timeSlots, timeSlot)
	}
	return timeSlots
}

func (s *TimeSlotService) getNextWeekStart(now time.Time) time.Time {
	daysUntilMonday := (8 - int(now.Weekday())) % 7
	if daysUntilMonday == 0 {
		daysUntilMonday = 7
	}
	nextMonday := now.AddDate(0, 0, daysUntilMonday)
	return time.Date(nextMonday.Year(), nextMonday.Month(), nextMonday.Day(), 0, 0, 0, 0, nextMonday.Location())
}

func (s *TimeSlotService) GetByDoctorAndMonth(doctorID string, date time.Time) ([]appointment.TimeSlot, error) {
	return s.timeSlotRepo.FindByDoctorAndMonth(doctorID, date)
}

func (s *TimeSlotService) GetByDoctorAndDate(doctorID string, date time.Time) ([]appointment.TimeSlot, error) {
	return s.timeSlotRepo.FindByDoctorAndDate(doctorID, date)
}

func (s *TimeSlotService) GetByDoctorAndDateRange(doctorID string, startDate, endDate time.Time) ([]appointment.TimeSlot, error) {
	return s.timeSlotRepo.FindByDoctorAndDateRange(doctorID, startDate, endDate)
}
	

func generateSlotID() string {
	return uuid.NewString()
}


func (s *TimeSlotService) CreateBatch(doctorID string, inputs []TimeSlotInput) ([]appointment.TimeSlot, error) {
	if doctorID == "" {
		return nil, errors.New("doctorID is required")
	}
	if len(inputs) == 0 {
		return nil, errors.New("no time slots provided")
	}

	var slots []appointment.TimeSlot
	now := time.Now()

	for _, in := range inputs {
		if in.Capacity <= 0 {
			return nil, errors.New("capacity must be greater than 0")
		}
		if !in.StartTime.Before(in.EndTime) {
			return nil, errors.New("startTime must be before endTime")
		}

		// check overlap
		count, err := s.timeSlotRepo.CountOverlapping(doctorID, in.StartTime, in.EndTime)
		if err != nil {
			return nil, fmt.Errorf("failed to check overlapping: %w", err)
		}
		if count > 0 {
			return nil, fmt.Errorf("doctor already has a timeslot in %v - %v", in.StartTime, in.EndTime)
		}

		slots = append(slots, appointment.TimeSlot{
			SlotID:    generateSlotID(),
			DoctorID:  doctorID,
			StartTime: in.StartTime,
			EndTime:   in.EndTime,
			Capacity:  in.Capacity,
			CreatedAt: now,
			UpdatedAt: now,
		})
	}

	if err := s.timeSlotRepo.CreateBatch(slots); err != nil {
		return nil, err
	}

	return slots, nil
}
// Định nghĩa khung giờ các ca
var shiftDefinitions = map[string][]struct {
	startHour, startMinute int
	endHour, endMinute     int
}{
	"morning": {
		{8, 30, 9, 0},   // 08:30 - 09:00
		{9, 0, 10, 0},   // 09:00 - 10:00
		{10, 0, 11, 0},  // 10:00 - 11:00
		{11, 0, 12, 0},  // 11:00 - 12:00
	},
	"afternoon": {
		{13, 30, 14, 0}, // 13:30 - 14:00
		{14, 0, 15, 0},  // 14:00 - 15:00
		{15, 0, 16, 0},  // 15:00 - 16:00
		{16, 0, 17, 0},  // 16:00 - 17:00
	},
	"evening": {
		{18, 0, 19, 0},  // 18:00 - 19:00
		{19, 0, 20, 0},  // 19:00 - 20:00
		{20, 0, 21, 0},  // 20:00 - 21:00
	},
}

func (s *TimeSlotService) CreateMultiShiftSlots(req CreateMultiShiftSlotsRequest) ([]appointment.TimeSlot, error) {
    var allSlots []appointment.TimeSlot
  	loc, _ := time.LoadLocation("Asia/Ho_Chi_Minh")
    for _, sel := range req.Shifts {
        date, err := time.Parse("2006-01-02", sel.Date)
        if err != nil {
            return nil, fmt.Errorf("invalid date: %s", sel.Date)
        }

        for _, shift := range sel.Slots {
            defs, ok := shiftDefinitions[shift]
            if !ok {
                return nil, fmt.Errorf("invalid shift: %s", shift)
            }

            for _, def := range defs {
                start := time.Date(date.Year(), date.Month(), date.Day(),
					def.startHour, def.startMinute, 0, 0, loc)
				end := time.Date(date.Year(), date.Month(), date.Day(),
     				def.endHour, def.endMinute, 0, 0, loc)             

                slot := appointment.TimeSlot{
                    SlotID:    generateSlotID(),
                    DoctorID:  req.DoctorID,
                    StartTime: start,
                    EndTime:   end,
                    Capacity:  1,
                    CreatedAt: time.Now(),
                    UpdatedAt: time.Now(),
                }
                allSlots = append(allSlots, slot)
            }
        }
    }

    if err := s.timeSlotRepo.CreateBatch(allSlots); err != nil {
        return nil, err
    }

    return allSlots, nil
}

func (s *TimeSlotService) ImportDoctorDayOff(filePath string) error {
	f, err := excelize.OpenFile(filePath)
	if err != nil {
		return fmt.Errorf("failed to open excel: %w", err)
	}

	sheetName := f.GetSheetName(0)
	rows, err := f.GetRows(sheetName)
	if err != nil {
		return fmt.Errorf("failed to read rows: %w", err)
	}

	// Bỏ qua header
	for i, row := range rows {
		if i == 0 {
			continue
		}

		if len(row) < 4 {
			continue
		}

		doctorID := strings.TrimSpace(row[0])
		dayOffList := strings.TrimSpace(row[2]) // "08/10/2025;10/10/2025"
		shiftList := strings.TrimSpace(row[3])  // "morning;afternoon" hoặc "all"

		if doctorID == "" || dayOffList == "" {
			continue
		}

		// Tách ngày nghỉ và buổi nghỉ
		days := strings.Split(dayOffList, ";")
		shifts := strings.Split(shiftList, ";")

		for idx, d := range days {
			d = strings.TrimSpace(d)
			if d == "" {
				continue
			}

			day, err := time.Parse("02/01/2006", d)
			if err != nil {
				fmt.Printf("❌ Lỗi parse ngày '%s': %v\n", d, err)
				continue
			}

			// Nếu shiftList ít hơn days -> mặc định all
			var shift string
			if idx < len(shifts) {
				shift = strings.ToLower(strings.TrimSpace(shifts[idx]))
			} else {
				shift = "all"
			}

			if shift == "all" {
				// Xoá toàn bộ slots trong ngày
				startOfDay := time.Date(day.Year(), day.Month(), day.Day(), 0, 0, 0, 0, time.Local)
				endOfDay := startOfDay.Add(24 * time.Hour)
				err = s.timeSlotRepo.DeleteByDoctorIDAndDateRange(doctorID, startOfDay, endOfDay)
				if err != nil {
					fmt.Printf("❌ Lỗi xoá slot bác sĩ %s ngày %s: %v\n", doctorID, d, err)
					continue
				}
				fmt.Printf("✅ Đã xoá toàn bộ lịch bác sĩ %s ngày %s\n", doctorID, d)
			} else {
				// Xoá đúng các slots theo shiftDefinitions
				if slots, ok := shiftDefinitions[shift]; ok {
					for _, sl := range slots {
						start := time.Date(day.Year(), day.Month(), day.Day(), sl.startHour, sl.startMinute, 0, 0, time.Local)
						end := time.Date(day.Year(), day.Month(), day.Day(), sl.endHour, sl.endMinute, 0, 0, time.Local)

						err = s.timeSlotRepo.DeleteByDoctorIDAndDateRange(doctorID, start, end)
						if err != nil {
							fmt.Printf("❌ Lỗi xoá slot bác sĩ %s ngày %s buổi %s (%v-%v): %v\n",
								doctorID, d, shift, start, end, err)
							continue
						}
					}
					fmt.Printf("✅ Đã xoá lịch bác sĩ %s ngày %s buổi %s\n", doctorID, d, shift)
				} else {
					fmt.Printf("⚠️ Buổi '%s' không hợp lệ tại dòng %d\n", shift, i+1)
				}
			}
		}
	}

	return nil
}

