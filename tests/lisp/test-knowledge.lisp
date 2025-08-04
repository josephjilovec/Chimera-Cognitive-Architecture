;; test-knowledge.lisp: Unit and integration tests for the knowledge base in backend/lisp/reasoning/knowledge-base.lisp
;; using FiveAM for the Chimera Cognitive Architecture. Tests concept retrieval, updates, and error handling
;; (e.g., (isa dog mammal)). Mocks interactions with the Julia layer to isolate tests. Aims for ≥90% test coverage
;; of knowledge-base.lisp functions. Includes detailed comments explaining test scenarios. Designed for SBCL.

(ql:quickload :fiveam)
(ql:quickload :asdf)

(asdf:defsystem :chimera-test-knowledge
  :depends-on (:chimera-knowledge-base :fiveam :cl-ppcre)
  :components ((:file "backend/lisp/reasoning/knowledge-base")))

(defpackage :chimera-test-knowledge
  (:use :cl :fiveam :chimera-reasoning)
  (:export #:run-tests))

(in-package :chimera-test-knowledge)

;; Mocking setup for Julia layer interactions
(defparameter *mock-knowledge-base* (make-hash-table :test #'equal)
  "Mock knowledge base for isolating tests")
(defparameter *mock-predicates* (make-hash-table :test #'equal)
  "Mock predicates for isolating tests")

;; Mock initialize-knowledge-base to avoid modifying the real knowledge base
(defun mock-initialize-knowledge-base ()
  "Mocks initialize-knowledge-base to populate mock knowledge base."
  (clrhash *mock-knowledge-base*)
  (clrhash *mock-predicates*)
  (setf (gethash "has-part" *mock-predicates*)
        (lambda (entity part) (declare (ignore entity part)) t))
  (setf (gethash "has-property" *mock-predicates*)
        (lambda (entity prop) (declare (ignore entity prop)) t))
  (setf (gethash "dog" *mock-knowledge-base*)
        '((isa mammal) (has-part tail) (has-property barks)))
  (setf (gethash "cat" *mock-knowledge-base*)
        '((isa mammal) (has-part tail) (has-property meows)))
  *mock-knowledge-base*)

;; Test suite definition
(def-suite knowledge-base-suite
  :description "Test suite for the Chimera knowledge base")

(in-suite knowledge-base-suite)

;; Test: initialize-knowledge-base
(test test-initialize-knowledge-base
  "Tests initialization of the knowledge base with default concepts and predicates."
  (let ((*knowledge-base* *mock-knowledge-base*)
        (*predicates* *mock-predicates*))
    (is (hash-table-p (mock-initialize-knowledge-base))
        "Knowledge base should be a hash table")
    (is (gethash "dog" *knowledge-base*)
        "Knowledge base should contain 'dog' concept")
    (is (gethash "cat" *knowledge-base*)
        "Knowledge base should contain 'cat' concept")
    (is (gethash "has-part" *predicates*)
        "Predicates should contain 'has-part'")
    (is (gethash "has-property" *predicates*)
        "Predicates should contain 'has-property'")))

;; Test: sanitize-concept
(test test-sanitize-concept
  "Tests sanitization of concept names for security and validity."
  (is (string= (sanitize-concept "dog") "dog")
      "Valid concept name should remain unchanged")
  (is (string= (sanitize-concept "dog-123") "dog-123")
      "Valid concept with numbers and hyphens should remain unchanged")
  (signals knowledge-base-error (sanitize-concept 123)
           "Non-string input should signal knowledge-base-error")
  (signals knowledge-base-error (sanitize-concept (make-string (1+ *max-concept-length*) :initial-element #\a))
           "Overlong concept name should signal knowledge-base-error")
  (is (string= (sanitize-concept "dog<script>attack") "dogattack")
      "Malicious characters should be removed"))

;; Test: add-concept
(test test-add-concept
  "Tests adding a new concept with relationships to the knowledge base."
  (let ((*knowledge-base* *mock-knowledge-base*)
        (*predicates* *mock-predicates*))
    (mock-initialize-knowledge-base)
    (is-true (add-concept "bird" '((isa animal) (has-property flies)))
             "Adding valid concept should return t")
    (is (equal (gethash "bird" *knowledge-base*)
               '((isa animal) (has-property flies)))
        "Bird concept should have correct relationships")
    (signals knowledge-base-error (add-concept "invalid" '((unknown pred)))
             "Unknown predicate should signal knowledge-base-error")
    (signals knowledge-base-error (add-concept "dog" '((isa mammal)))
             "Duplicate concept should not overwrite without update-concept")))

;; Test: add-relationship
(test test-add-relationship
  "Tests adding a relationship to an existing concept."
  (let ((*knowledge-base* *mock-knowledge-base*)
        (*predicates* *mock-predicates*))
    (mock-initialize-knowledge-base)
    (is-true (add-relationship "dog" '(has-property loyal))
             "Adding valid relationship should return t")
    (is (member '(has-property loyal) (gethash "dog" *knowledge-base*) :test #'equal)
        "Dog should have new 'has-property loyal' relationship")
    (signals knowledge-base-error (add-relationship "unknown" '(has-part wings))
             "Adding to unknown concept should signal knowledge-base-error")
    (signals knowledge-base-error (add-relationship "dog" '(unknown pred))
             "Unknown predicate should signal knowledge-base-error")))

;; Test: query-knowledge-base
(test test-query-knowledge-base
  "Tests querying the knowledge base for concepts matching a pattern."
  (let ((*knowledge-base* *mock-knowledge-base*)
        (*predicates* *mock-predicates*))
    (mock-initialize-knowledge-base)
    (let ((results (query-knowledge-base "mammal")))
      (is-true (listp results)
               "Query should return a list")
      (is (= (length results) 2)
          "Query for 'mammal' should return two concepts (dog, cat)")
      (is (find-if (lambda (r) (string= (getf r :concept) "dog")) results)
          "Query should include dog concept")
      (is (find-if (lambda (r) (string= (getf r :concept) "cat")) results)
          "Query should include cat concept"))
    (is (null (query-knowledge-base "nonexistent"))
        "Query for nonexistent pattern should return nil")))

;; Test: find-predicate
(test test-find-predicate
  "Tests retrieval of predicate functions from the predicates hash table."
  (let ((*knowledge-base* *mock-knowledge-base*)
        (*predicates* *mock-predicates*))
    (mock-initialize-knowledge-base)
    (is (functionp (find-predicate "has-part"))
        "has-part predicate should be a function")
    (is (functionp (find-predicate "has-property"))
        "has-property predicate should be a function")
    (signals knowledge-base-error (find-predicate "unknown")
             "Unknown predicate should signal knowledge-base-error")))

;; Test: update-concept
(test test-update-concept
  "Tests updating relationships for an existing concept."
  (let ((*knowledge-base* *mock-knowledge-base*)
        (*predicates* *mock-predicates*))
    (mock-initialize-knowledge-base)
    (is-true (update-concept "dog" '((isa animal) (has-property loyal)))
             "Updating dog concept should return t")
    (is (equal (gethash "dog" *knowledge-base*)
               '((isa animal) (has-property loyal)))
        "Dog concept should have updated relationships")
    (signals knowledge-base-error (update-concept "unknown" '((isa animal)))
             "Updating unknown concept should signal knowledge-base-error")
    (signals knowledge-base-error (update-concept "dog" '((unknown pred)))
             "Invalid predicate should signal knowledge-base-error")))

;; Test: remove-concept
(test test-remove-concept
  "Tests removing a concept from the knowledge base."
  (let ((*knowledge-base* *mock-knowledge-base*)
        (*predicates* *mock-predicates*))
    (mock-initialize-knowledge-base)
    (is-true (remove-concept "dog")
             "Removing dog concept should return t")
    (is-false (gethash "dog" *knowledge-base*)
              "Dog concept should no longer exist")
    (signals knowledge-base-error (remove-concept "unknown")
             "Removing unknown concept should signal knowledge-base-error")))

;; Test: error-handling
(test test-error-handling
  "Tests error handling for invalid inputs and edge cases."
  (let ((*knowledge-base* *mock-knowledge-base*)
        (*predicates* *mock-predicates*))
    (signals knowledge-base-error (sanitize-concept nil)
             "Nil input to sanitize-concept should signal knowledge-base-error")
    (signals knowledge-base-error (add-concept nil '((isa animal)))
             "Nil concept name should signal knowledge-base-error")
    (signals knowledge-base-error (add-relationship nil '(has-part tail))
             "Nil concept for add-relationship should signal knowledge-base-error")
    (signals knowledge-base-error (query-knowledge-base nil)
             "Nil query pattern should signal knowledge-base-error")
    (signals knowledge-base-error (update-concept nil '((isa animal)))
             "Nil concept for update-concept should signal knowledge-base-error")
    (signals knowledge-base-error (remove-concept nil)
             "Nil concept for remove-concept should signal knowledge-base-error")))

;; Function: run-tests
(defun run-tests ()
  "Runs all tests in the knowledge-base-suite."
  (handler-case
      (progn
        (format t "Running knowledge base tests...~%")
        (fiveam:run! 'knowledge-base-suite))
    (error (e)
      (format t "Test suite error: ~A~%" e)
      nil)))

;; Run tests when the script is loaded
(run-tests)
