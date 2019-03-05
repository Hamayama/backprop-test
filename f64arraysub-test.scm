;;;
;;; Test f64arraysub
;;;

(add-load-path "." :relative)
(use gauche.test)
(use gauche.array)
(use gauche.uvector)

(define (nearly=? x y :optional (precision 1e-12))
  (<= (abs (- x y)) precision))

(test-start "eigenmat")
(use f64arraysub)
(test-module 'eigenmat)

(define A (f64array-simple 0 2 0 2 1 2 3 4))
(define B (f64array-simple 0 2 0 2 5 6 7 8))
(define C (f64array-simple 0 2 0 2 1 1 1 1))
(define F (f64array-simple 0 2 0 2 0 0 0 0))
(define L (f64array-simple 0 2 0 3 -2 -1 0 1 2 3))

(define (run-test)
  (test* "f64array-ref 1" 1
         (f64array-ref A  0  0) nearly=?)
  (test* "f64array-ref 2" 4
         (f64array-ref A  1  1) nearly=?)
  (test* "f64array-ref 3" (test-error <error>)
         (f64array-ref A -1  0))
  (test* "f64array-ref 4" (test-error <error>)
         (f64array-ref A  0  3))

  (let1 A1 (f64array-copy A)
    (test* "f64array-set! 1" 100
          (begin (f64array-set! A1  0  0  100) (f64array-ref  A1  0  0)) nearly=?)
    (test* "f64array-set! 2" 400
          (begin (f64array-set! A1  1  1  400) (f64array-ref  A1  1  1)) nearly=?)
    (test* "f64array-set! 3" (test-error <error>)
          (f64array-set! A1 -1  0  100))
    (test* "f64array-set! 4" (test-error <error>)
          (f64array-set! A1  0  3  400))
    )

  (test* "f64array-copy 1" A
         (f64array-copy A)  f64array-nearly=?)

  (test* "f64array-map 1"  B
         (f64array-map (lambda (d1) (+ d1 4)) A) f64array-nearly=?)

  (test* "make-f64array-simple 1" F
         (make-f64array-simple 0 2 0 2) f64array-nearly=?)

  (test* "f64array-simple 1" A
         (f64array-simple 0 2 0 2 1 2 3 4) f64array-nearly=?)

  (test* "f64array-nearly=? 1" #t
         (f64array-nearly=? A (f64array-simple 0 2 0 2 1 2 3 4)))
  (test* "f64array-nearly=? 2" #t
         (f64array-nearly=? A (f64array-simple 0 2 0 2 1 2 3 (+ 4 1e-13))))
  (test* "f64array-nearly=? 3" #f
         (f64array-nearly=? A (f64array-simple 0 2 0 2 1 2 3 (+ 4 1e-11))))
  (test* "f64array-nearly=? 4" #f
         (f64array-nearly=? F (f64array-simple 0 2 0 2 0 0 0 1e-13)))

  (test* "f64array-nearly-zero? 1" #t
         (f64array-nearly-zero? F))
  (test* "f64array-nearly-zero? 2" #t
         (f64array-nearly-zero? (f64array-simple 0 2 0 2 0 0 0 1e-13)))
  (test* "f64array-nearly-zero? 3" #f
         (f64array-nearly-zero? (f64array-simple 0 2 0 2 0 0 0 1e-11)))

  (let ((A1 (f64array-copy A))
        (L1 (f64array-copy L)))
    (define (sigmoid x) (/. 1 (+ 1 (exp (- x)))))

    (test* "f64array-add-elements 1"  #,(<f64array> (0 2 0 2) 3 6 9 12)
           (f64array-add-elements A A A) f64array-nearly=?)

    (test* "f64array-add-elements! 1" #,(<f64array> (0 2 0 2) 3 6 9 12)
           (begin (f64array-add-elements! A1 A A A) A1) f64array-nearly=?)

    (test* "f64array-sub-elements 1"  #,(<f64array> (0 2 0 2) -1 -2 -3 -4)
           (f64array-sub-elements A A A) f64array-nearly=?)

    (test* "f64array-sub-elements! 1" #,(<f64array> (0 2 0 2) -1 -2 -3 -4)
           (begin (f64array-sub-elements! A1 A A A) A1) f64array-nearly=?)

    (test* "f64array-mul 1"  #,(<f64array> (0 2 0 2) 7 10 15 22)
           (f64array-mul A A) f64array-nearly=?)

    (test* "f64array-mul! 1" #,(<f64array> (0 2 0 2) 7 10 15 22)
           (begin (f64array-mul! A1 A A) A1) f64array-nearly=?)

    (test* "f64array-mul-elements 1"  #,(<f64array> (0 2 0 2) 1 8 27 64)
           (f64array-mul-elements A A A) f64array-nearly=?)

    (test* "f64array-mul-elements! 1" #,(<f64array> (0 2 0 2) 1 8 27 64)
           (begin (f64array-mul-elements! A1 A A A) A1) f64array-nearly=?)

    (test* "f64array-sigmoid 1"  (f64array-simple
                                  0 2 0 2
                                  (sigmoid 5) (sigmoid 6) (sigmoid 7) (sigmoid 8))
           (f64array-sigmoid B) f64array-nearly=?)

    (test* "f64array-sigmoid! 1" (f64array-simple
                                  0 2 0 2
                                  (sigmoid 5) (sigmoid 6) (sigmoid 7) (sigmoid 8))
           (f64array-sigmoid! A1 B) f64array-nearly=?)

    (test* "f64array-relu 1"  #,(<f64array> (0 2 0 3) 0 0 0 1 2 3)
           (f64array-relu L) f64array-nearly=?)

    (test* "f64array-relu! 1" #,(<f64array> (0 2 0 3) 0 0 0 1 2 3)
           (f64array-relu! L1 L) f64array-nearly=?)

    (test* "f64array-step 1"  #,(<f64array> (0 2 0 3) 0 0 0 1 1 1)
           (f64array-step L) f64array-nearly=?)

    (test* "f64array-step! 1" #,(<f64array> (0 2 0 3) 0 0 0 1 1 1)
           (f64array-step! L1 L) f64array-nearly=?)
    )

  (test* "f64array-transpose 1" #,(<f64array> (0 3 0 2) -2 1 -1 2 0 3)
         (f64array-transpose L) f64array-nearly=?)

  (test* "f64array-row 1" #,(<f64array> (0 1 0 3) 1 2 3)
         (f64array-row L 1) f64array-nearly=?)

  (test* "f64array-col 1" #,(<f64array> (0 2 0 1) 0 3)
         (f64array-col L 2) f64array-nearly=?)
  )

(test-section "use eigenmat")
(run-test)

(test-section "don't use eigenmat")
(select-module f64arraysub)
(define *disable-eigenmat* #t)
(select-module user)
(load "f64arraysub")
(run-test)

(format (current-error-port) "~%~a" ((with-module gauche.test format-summary)))

;; If you don't want `gosh' to exit with nonzero status even if
;; the test fails, pass #f to :exit-on-failure.
(test-end :exit-on-failure #t)

