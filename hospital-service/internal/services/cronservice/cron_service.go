package cronservice

import (
	"hospital-service/internal/services/appointmentservice"
	"log"
	"time"

	"github.com/robfig/cron/v3"
)

type CronService struct {
	timeSlotService *appointmentservice.TimeSlotService
	cron            *cron.Cron
}

func NewCronService(timeSlotService *appointmentservice.TimeSlotService) *CronService {
	// Tạo cron với timezone Việt Nam
	loc, _ := time.LoadLocation("Asia/Ho_Chi_Minh")
	c := cron.New(cron.WithLocation(loc))

	return &CronService{
		timeSlotService: timeSlotService,
		cron:            c,
	}
}

// Start khởi động cron service
func (s *CronService) Start() error {
	// Thêm job chạy vào ngày 28 hàng tháng lúc 00:00 để tạo lịch cho tháng tiếp theo
	// Cron expression: "0 0 28 * *" (phút giờ ngày tháng thứ)
	_, err := s.cron.AddFunc("0 0 28 * *", s.generateMonthlyTimeSlots)
	if err != nil {
		log.Printf("Error adding cron job: %v", err)
		return err
	}

	s.cron.Start()
	log.Println("Cron service started successfully - Will run every 28th of the month at 00:00")
	return nil
}

// Stop dừng cron service
func (s *CronService) Stop() {
	s.cron.Stop()
	log.Println("Cron service stopped")
}

// generateMonthlyTimeSlots job chính để generate time slots hàng tháng
func (s *CronService) generateMonthlyTimeSlots() {
	log.Println("🚀 Starting monthly time slots generation...")
	
	startTime := time.Now()
	err := s.timeSlotService.GenerateTimeSlotsForMonth()
	duration := time.Since(startTime)
	
	if err != nil {
		log.Printf("❌ Error generating monthly time slots: %v (took %v)", err, duration)
	} else {
		log.Printf("✅ Successfully generated monthly time slots (took %v)", duration)
	}
}

// testJob job test để kiểm tra cron hoạt động
func (s *CronService) testJob() {
	log.Println("Test cron job executed at", time.Now().Format("2006-01-02 15:04:05"))
}

// immediateTestJob job test chạy mỗi 30 giây để debug
func (s *CronService) immediateTestJob() {
	log.Println("IMMEDIATE TEST: Cron is working! Time:", time.Now().Format("2006-01-02 15:04:05"))
}

// GetCronEntries trả về danh sách các cron entries đang chạy
func (s *CronService) GetCronEntries() []cron.Entry {
	return s.cron.Entries()
}
