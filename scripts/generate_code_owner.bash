# bash syntax function for current directory git repository
owners(){
  for f in $(git ls-files); do
    # filename
    echo -n "$f "
    # author emails if loc distribution >= 30%
    git fame --show=name,email -snwMC --incl "$f" | tr '/' '|' \
      | awk -F '|' '(NR>6 && $6>=30) {print $2}' \
      | xargs echo
  done
}

# print to screen and file
owners | tee .github/CODEOWNERS

# same but with `tqdm` progress for large repos
owners \
  | tqdm --total $(git ls-files | wc -l) \
    --unit file --desc "Generating CODEOWNERS" \
  > ../.github/CODEOWNERS
